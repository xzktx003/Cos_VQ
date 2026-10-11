import pytest
import torch

from experiments.nowag_d6.finetune_hard_codebook_e2e import validate_exported_state


def state():
    return dict(assignments=torch.tensor([0, 1], dtype=torch.int32),
                codebook=torch.zeros(2, 2, dtype=torch.float16),
                row_norm=torch.ones(1, dtype=torch.float16),
                col_norm=torch.ones(4, dtype=torch.float16),
                shape=(1, 4), vector_dim=2, assignment_bits=1)


def test_changed_assignment_only_allowed_in_new_mode():
    original = state()
    changed = state()
    changed['assignments'][0] = 1
    with pytest.raises(ValueError):
        validate_exported_state(changed, original, False)
    validate_exported_state(changed, original, True)


@pytest.mark.parametrize('key,value', [
    ('assignments', torch.tensor([0, 2], dtype=torch.int32)),
    ('assignments', torch.tensor([0, 1], dtype=torch.int64)),
    ('assignment_bits', 2), ('shape', (2, 2)),
    ('codebook', torch.zeros(3, 2, dtype=torch.float16)),
])
def test_changed_assignment_mode_preserves_budget(key, value):
    original = state()
    changed = state()
    changed[key] = value
    with pytest.raises(ValueError):
        validate_exported_state(changed, original, True)


def test_binary_llama_checkpoint_gradients_and_export():
    import copy
    from transformers import LlamaConfig, LlamaForCausalLM
    from experiments.nowag_d6.finetune_hard_codebook_e2e import HardCodebookLinear
    from experiments.nowag_d6.logical_state import reconstruct_logical_state

    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).requires_grad_(False)
    dense = copy.deepcopy(model)
    for name, linear in list(model.named_modules()):
        if isinstance(linear, torch.nn.Linear) and name != 'lm_head':
            rows, columns = linear.weight.shape
            source = dict(shape=torch.tensor([rows, columns]), vector_dim=torch.tensor(2),
                          assignment_bits=torch.tensor(2),
                          assignments=torch.randint(4, (rows * columns // 2,)),
                          codebook=torch.randn(4, 2).half(),
                          row_norm=torch.full((rows,), 0.02).half(),
                          col_norm=torch.ones(columns).half())
            parent, attr = name.rsplit('.', 1)
            replacement = HardCodebookLinear(linear, source, (source['assignments'] + 1) % 4)
            setattr(model.get_submodule(parent), attr, replacement)
    recomputed = copy.deepcopy(model)
    recomputed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.randint(32, (1, 8))
    for candidate in (model, recomputed):
        candidate.train()
        candidate(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    for (name, param), (other_name, other) in zip(model.named_parameters(), recomputed.named_parameters()):
        assert name == other_name
        if param.requires_grad:
            assert param.grad is not None
            torch.testing.assert_close(param.grad, other.grad)
            if name.endswith('assignment_logits'):
                assert param.grad.abs().sum() > 0
    for name, module in model.named_modules():
        if isinstance(module, HardCodebookLinear):
            with torch.no_grad():
                module.quantizer.assignment_logits[::2] = 0.1
            exported = module.quantizer.export_state()
            dense.get_submodule(name).weight.data.copy_(reconstruct_logical_state(exported, torch.device('cpu')))
    model.eval()
    dense.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, dense(ids).logits, rtol=0, atol=0)
