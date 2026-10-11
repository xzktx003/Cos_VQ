import copy

import pytest
import torch
from torch.nn import functional as F

from experiments.nowag_d6.finetune_hard_codebook_e2e import distillation_kl


def reference(student, teacher):
    s = student[:, :-1].float().log_softmax(-1)
    t = teacher[:, :-1].detach().float().log_softmax(-1)
    return F.kl_div(s, t, log_target=True, reduction='sum') / s.shape[0] / s.shape[1]


def test_identical_logits_have_zero_kl():
    logits = torch.randn(2, 5, 7, requires_grad=True)
    loss = distillation_kl(logits, logits, chunk_size=2)
    assert loss.item() == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize('chunk_size', [1, 3, 20])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_chunked_kl_matches_reference_and_student_gradient(chunk_size, dtype):
    torch.manual_seed(8)
    student = torch.randn(2, 8, 13, dtype=dtype, requires_grad=True)
    teacher = torch.randn(2, 8, 13, dtype=dtype, requires_grad=True)
    actual = distillation_kl(student, teacher, chunk_size=chunk_size)
    expected = reference(student, teacher)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    observed = student.grad.clone()
    assert teacher.grad is None
    student.grad = None
    expected.backward()
    torch.testing.assert_close(observed, student.grad)
    assert torch.count_nonzero(observed[:, -1]) == 0
    assert actual.dtype == torch.float32


def test_kl_direction_is_teacher_to_student():
    student = torch.tensor([[[3., 0., -1.], [0., 0., 0.]]])
    teacher = torch.tensor([[[0., 1., 0.], [0., 0., 0.]]])
    result = distillation_kl(student, teacher)
    torch.testing.assert_close(result, reference(student, teacher))
    assert abs(float(result - reference(teacher, student))) > 0.1


@pytest.mark.parametrize('shape,chunk', [((2, 1, 4), 2), ((2, 3, 4), 0)])
def test_invalid_inputs_rejected(shape, chunk):
    with pytest.raises(ValueError):
        distillation_kl(torch.zeros(shape), torch.zeros(shape), chunk_size=chunk)


def test_tiny_llama_distillation_checkpoint_gradients_and_export():
    from transformers import LlamaConfig, LlamaForCausalLM
    from experiments.nowag_d6.finetune_hard_codebook_e2e import HardCodebookLinear
    from experiments.nowag_d6.logical_state import LINEAR_NAMES, reconstruct_logical_state

    torch.manual_seed(4)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    teacher = LlamaForCausalLM(config).requires_grad_(False).eval()
    student = copy.deepcopy(teacher)
    dense = copy.deepcopy(teacher)
    for block in student.model.layers:
        for name in LINEAR_NAMES:
            parent_name, attribute = name.rsplit('.', 1)
            parent = block.get_submodule(parent_name)
            linear = getattr(parent, attribute)
            rows, columns = linear.weight.shape
            state = {'shape': torch.tensor([rows, columns]), 'vector_dim': torch.tensor(2),
                     'assignment_bits': torch.tensor(2),
                     'assignments': torch.randint(4, (rows * columns // 2,)),
                     'codebook': torch.randn(4, 2).half(),
                     'row_norm': torch.full((rows,), 0.02).half(),
                     'col_norm': torch.ones(columns).half()}
            setattr(parent, attribute, HardCodebookLinear(linear, state))
    recomputed = copy.deepcopy(student)
    recomputed.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ids = torch.randint(32, (1, 8))
    with torch.no_grad():
        target = teacher(ids, use_cache=False).logits
    for candidate in (student, recomputed):
        candidate.train()
        distillation_kl(candidate(ids, use_cache=False).logits, target, chunk_size=3).backward()
    assert all(p.grad is None for p in teacher.parameters())
    for (name, parameter), (other_name, other) in zip(student.named_parameters(), recomputed.named_parameters()):
        assert name == other_name
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
            torch.testing.assert_close(parameter.grad, other.grad)
    torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=1e-3).step()
    for name, module in student.named_modules():
        if isinstance(module, HardCodebookLinear):
            state = module.quantizer.export_state()
            with torch.no_grad():
                dense.get_submodule(name).weight.copy_(reconstruct_logical_state(state, torch.device('cpu')))
    student.eval()
    dense.eval()
    with torch.no_grad():
        torch.testing.assert_close(student(ids).logits, dense(ids).logits, rtol=0, atol=0)
