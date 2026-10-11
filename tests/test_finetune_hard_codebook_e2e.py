import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from experiments.nowag_d6.finetune_hard_codebook_e2e import HardCodebookLinear


@pytest.mark.parametrize('suffix', ['checkpoint.pt', 'checkpoint.pt.tmp', 'progress.json', 'progress.json.tmp'])
@pytest.mark.parametrize('kind', ['existing', 'same_path', 'symlink', 'hardlink', 'dangling_symlink'])
def test_output_preflight_preserves_existing_files(tmp_path, suffix, kind):
    from experiments.nowag_d6.finetune_hard_codebook_e2e import validate_output_paths
    root = tmp_path / 'run'
    root.mkdir()
    source = tmp_path / 'source.pt'
    source.write_bytes(b'original checkpoint')
    target = root / f'20260907_120000_{suffix}'
    if kind == 'same_path':
        source.rename(target)
        source = target
    elif kind == 'symlink':
        target.symlink_to(source)
    elif kind == 'hardlink':
        target.hardlink_to(source)
    elif kind == 'dangling_symlink':
        target.symlink_to(tmp_path / 'missing')
    else:
        target.write_bytes(b'previous run')
    before = source.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        validate_output_paths(source, root, '20260907_120000')
    assert source.read_bytes() == before
    assert target.is_symlink() or target.exists()
    if kind == 'existing':
        assert target.read_bytes() == b'previous run'


def test_output_preflight_accepts_fresh_directory_without_writes(tmp_path):
    from experiments.nowag_d6.finetune_hard_codebook_e2e import validate_output_paths
    source = tmp_path / 'source.pt'
    source.write_bytes(b'original checkpoint')
    root = tmp_path / 'new_run'
    checkpoint_path, progress_path = validate_output_paths(source, root, '20260907_120000')
    assert checkpoint_path == root / '20260907_120000_checkpoint.pt'
    assert progress_path == root / '20260907_120000_progress.json'
    assert not root.exists()


def test_hard_linear_backward_through_checkpoint_with_frozen_inputs():
    state = {'shape': torch.tensor([2, 2]), 'vector_dim': torch.tensor(2),
             'assignment_bits': torch.tensor(1), 'assignments': torch.tensor([0, 1]),
             'codebook': torch.eye(2).half(), 'row_norm': torch.ones(2).half(),
             'col_norm': torch.ones(2).half()}
    base = nn.Linear(2, 2, bias=False)
    layer = HardCodebookLinear(base, state)
    x = torch.ones(1, 2)
    output = checkpoint(layer, x, use_reentrant=False)
    output.square().mean().backward()
    assert layer.quantizer.codebook.grad.abs().sum() > 0
    assert x.grad is None
    assert torch.equal(layer(x).detach(), x)
    assert set(dict(layer.named_parameters())) == {
        'quantizer.codebook', 'quantizer.row_norm_delta', 'quantizer.col_norm_delta'}


def test_two_layer_qwen_checkpoint_gradients_and_fresh_export_logits():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from experiments.nowag_d6.logical_state import reconstruct_logical_state
    torch.manual_seed(1)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).requires_grad_(False)
    dense = copy.deepcopy(model)
    for block in model.model.layers:
        for parent_name, names in [('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
                                   ('mlp', ('gate_proj', 'up_proj', 'down_proj'))]:
            parent = getattr(block, parent_name)
            for name in names:
                linear = getattr(parent, name)
                rows, columns = linear.weight.shape
                state = {'shape': torch.tensor([rows, columns]), 'vector_dim': torch.tensor(2),
                         'assignment_bits': torch.tensor(2),
                         'assignments': torch.randint(4, (rows * columns // 2,)),
                         'codebook': torch.randn(4, 2).half(),
                         'row_norm': torch.full((rows,), 0.02).half(),
                         'col_norm': torch.ones(columns).half()}
                setattr(parent, name, HardCodebookLinear(linear, state))
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
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    for name, module in model.named_modules():
        if isinstance(module, HardCodebookLinear):
            state = module.quantizer.export_state()
            dense.get_submodule(name).weight.data.copy_(reconstruct_logical_state(state, torch.device('cpu')))
    model.eval()
    dense.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, dense(ids).logits, rtol=0, atol=0)
