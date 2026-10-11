"""CPU checks against real Transformers Llama and Qwen3 decoder forwards."""

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    modeling_utils,
)

from experiments.nowag_d6.logical_state import get_linear_modules
from experiments.nowag_d6.v19_model_driver import (
    capture_first_layer_inputs,
    collect_block_hessians,
    move_tree,
    propagate_block,
)
from experiments.qwen_qtip.qtip_runtime import install_tp_plan_compat


def tiny_model(family, dtype):
    config_cls, model_cls = {
        'llama': (LlamaConfig, LlamaForCausalLM),
        'qwen3': (Qwen3Config, Qwen3ForCausalLM),
    }[family]
    config = config_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = 'sdpa'
    # Torch 2.4 / Transformers 4.52 needs the same production loader repair.
    if getattr(modeling_utils, 'ALL_PARALLEL_STYLES', ()) is None:
        install_tp_plan_compat(config)
    return model_cls(config).to(dtype=dtype).requires_grad_(False).eval()


def assert_tree_equal(observed, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert observed.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(observed[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(observed) == len(expected)
        for first, second in zip(observed, expected):
            assert_tree_equal(first, second)
    else:
        assert observed == expected


@pytest.mark.parametrize('family', ['llama', 'qwen3'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_real_transformers_capture_and_block_propagation_match_model(family, dtype):
    torch.manual_seed(19)
    model = tiny_model(family, dtype)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    captured, block_outputs = [], {0: [], 1: []}

    def first_input(_module, args, kwargs):
        captured.append((args[0].detach().clone(), move_tree(kwargs, 'cpu')))

    def block_output(index):
        def capture(_module, _args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            block_outputs[index].append(hidden.detach().clone())
        return capture

    handles = [model.model.layers[0].register_forward_pre_hook(first_input, with_kwargs=True)]
    handles.extend(block.register_forward_hook(block_output(i))
                   for i, block in enumerate(model.model.layers))
    try:
        with torch.no_grad():
            for row in tokens:
                model(input_ids=row[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    original_first = model.model.layers[0]
    batches = capture_first_layer_inputs(model, tokens, torch.device('cpu'))
    assert model.model.layers[0] is original_first
    assert_tree_equal(batches, captured)
    assert 'position_embeddings' in batches[0][1]
    for index, block in enumerate(model.model.layers):
        batches = propagate_block(block, batches, torch.device('cpu'))
        for (hidden, _kwargs), expected in zip(batches, block_outputs[index]):
            torch.testing.assert_close(hidden, expected, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['llama', 'qwen3'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_hessians_follow_modified_prefix_and_stay_fixed_during_block_updates(family, dtype):
    torch.manual_seed(23)
    model = tiny_model(family, dtype)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    modules = get_linear_modules(model, 1)
    with torch.no_grad():
        # Stand in for an accepted prefix update and the source quantized block.
        model.model.layers[0].self_attn.v_proj.weight.mul_(0.25)
        model.model.layers[0].mlp.down_proj.weight.mul_(1.75)
        for module in modules.values():
            module.weight.mul_(0.5)

    reference_inputs = {name: [] for name in modules}

    def linear_input(name):
        def capture(_module, args):
            reference_inputs[name].append(args[0].detach().reshape(-1, args[0].shape[-1]).double().clone())
        return capture

    handles = [module.register_forward_pre_hook(linear_input(name))
               for name, module in modules.items()]
    try:
        with torch.no_grad():
            for row in tokens:
                model(input_ids=row[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    batches = capture_first_layer_inputs(model, tokens, torch.device('cpu'))
    batches = propagate_block(model.model.layers[0], batches, torch.device('cpu'))
    hessians, counts = collect_block_hessians(model.model.layers[1], modules, batches, torch.device('cpu'))
    snapshots = {name: value.clone() for name, value in hessians.items()}
    for name, values in reference_inputs.items():
        inputs = torch.cat(values)
        expected = inputs.T @ inputs / len(inputs)
        torch.testing.assert_close(hessians[name], expected, rtol=1e-12, atol=1e-12)
        assert counts[name] == tokens.numel()
        assert not modules[name]._forward_pre_hooks

    with torch.no_grad():
        modules['mlp.down_proj'].weight.zero_()
        propagate_block(model.model.layers[1], batches, torch.device('cpu'))
    for name in hessians:
        torch.testing.assert_close(hessians[name], snapshots[name], rtol=0, atol=0)
