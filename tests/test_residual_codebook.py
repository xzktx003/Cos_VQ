import copy

import pytest
import torch
from torch import nn

from experiments.nowag_d6.hard_codebook import HardCodebookWeight
from experiments.nowag_d6.residual_codebook import ResidualCodebookWeight
from experiments.nowag_d6.finetune_hard_codebook_e2e import HardCodebookLinear, validate_exported_state


def state(rows=4, columns=4):
    return {'shape': torch.tensor([rows, columns]), 'vector_dim': torch.tensor(2),
            'assignment_bits': torch.tensor(2),
            'assignments': torch.randint(4, (rows * columns // 2,)),
            'codebook': torch.randn(4, 2).half(),
            'row_norm': torch.full((rows,), .02).half(),
            'col_norm': torch.ones(columns).half()}


def test_zero_initialization_gradients_export_and_reload():
    original = state()
    module = ResidualCodebookWeight(original, rank=2)
    assert torch.equal(module(), HardCodebookWeight(original)())
    module().square().sum().backward()
    assert module.residual_B.grad.abs().sum() > 0
    assert torch.count_nonzero(module.residual_A.grad) == 0
    with torch.no_grad():
        module.residual_B.normal_(std=.02)
    module.zero_grad()
    module().square().sum().backward()
    assert module.residual_A.grad.abs().sum() > 0
    exported = module.export_state()
    validate_exported_state(exported, original, residual_rank=2)
    expected = HardCodebookWeight(original)() + exported['residual_B'].float() @ exported['residual_A'].float()
    assert torch.equal(module(), expected)
    assert torch.equal(ResidualCodebookWeight(exported, rank=2)(), module())
    assert module.residual_bits == 16 * 2 * (4 + 4)
    with pytest.raises(ValueError):
        HardCodebookLinear(nn.Linear(4, 4), exported)


def test_existing_factors_cannot_be_silently_dropped_or_reshaped():
    exported = ResidualCodebookWeight(state(), rank=2).export_state()
    with pytest.raises(ValueError):
        ResidualCodebookWeight(exported, rank=1)
    del exported['residual_A']
    with pytest.raises(ValueError):
        ResidualCodebookWeight(exported, rank=2)


@pytest.mark.parametrize('option', ['--binary-assignments', '--distillation'])
def test_residual_cli_rejects_other_experimental_modes(monkeypatch, option):
    from experiments.nowag_d6.finetune_hard_codebook_e2e import main
    monkeypatch.setattr('sys.argv', ['train', '--model', 'unused', '--checkpoint', 'unused',
                                    '--token-cache', 'unused', '--output-dir', 'unused',
                                    '--residual-rank', '2', option])
    with pytest.raises(ValueError, match='unsupported'):
        main()


@pytest.mark.parametrize('corruption', ['extra', 'dtype', 'shape', 'finite', 'assignment'])
def test_residual_budget_rejects_corruption(corruption):
    original = state()
    exported = ResidualCodebookWeight(original, rank=2).export_state()
    if corruption == 'extra':
        exported['unbudgeted'] = torch.ones(1)
    elif corruption == 'dtype':
        exported['residual_A'] = exported['residual_A'].float()
    elif corruption == 'shape':
        exported['residual_B'] = torch.zeros(4, 3).half()
    elif corruption == 'finite':
        exported['residual_A'][0, 0] = float('inf')
    else:
        exported['assignments'][0] = (exported['assignments'][0] + 1) % 4
    with pytest.raises(ValueError):
        validate_exported_state(exported, original, residual_rank=2)


def test_tiny_llama_checkpoint_gradients_and_export_logits():
    from transformers import LlamaConfig, LlamaForCausalLM
    from experiments.nowag_d6.logical_state import LINEAR_NAMES, reconstruct_logical_state
    torch.manual_seed(13)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).requires_grad_(False)
    dense = copy.deepcopy(model)
    for block in model.model.layers:
        for name in LINEAR_NAMES:
            parent_name, attr = name.rsplit('.', 1)
            parent = block.get_submodule(parent_name)
            linear = getattr(parent, attr)
            replacement = HardCodebookLinear(linear, state(*linear.weight.shape), residual_rank=2)
            with torch.no_grad():
                replacement.quantizer.residual_B.normal_(std=.01)
            setattr(parent, attr, replacement)
    recomputed = copy.deepcopy(model)
    recomputed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.randint(32, (1, 8))
    for candidate in (model, recomputed):
        candidate.train()
        candidate(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), recomputed.named_parameters()):
        assert name == other_name
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
            torch.testing.assert_close(parameter.grad, other.grad)
    torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001).step()
    for name, module in model.named_modules():
        if isinstance(module, HardCodebookLinear):
            exported = module.quantizer.export_state()
            dense.get_submodule(name).weight.data.copy_(reconstruct_logical_state(exported, torch.device('cpu')))
    model.eval()
    dense.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, dense(ids).logits, rtol=0, atol=0)
