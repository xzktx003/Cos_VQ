"""CPU contracts for optional resident multi-GPU evaluation placement."""

from types import SimpleNamespace

import pytest
import torch

from experiments.nowag_d6 import v19_model_placement as placement


def tiny_model(family="llama", layers=4):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM, modeling_utils
    config_class, model_class = (LlamaConfig, LlamaForCausalLM) if family == "llama" else (Qwen3Config, Qwen3ForCausalLM)
    config = config_class(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=layers, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
                          tie_word_embeddings=False)
    if getattr(modeling_utils, "ALL_PARALLEL_STYLES", ()) is None:
        from experiments.qwen_qtip.qtip_runtime import install_tp_plan_compat
        install_tp_plan_compat(config)
    return model_class(config).eval()


def checkpoint_for(model):
    from experiments.nowag_d6.logical_state import get_linear_modules
    layers = {}
    for index in range(len(model.model.layers)):
        layers[str(index)] = {}
        for name, module in get_linear_modules(model, index).items():
            rows, columns = module.weight.shape
            layers[str(index)][name] = {
                "shape": (rows, columns), "vector_dim": 1, "assignment_bits": 1,
                "assignments": torch.zeros(rows, columns, dtype=torch.int16),
                "codebook": torch.tensor([[.25], [.5]], dtype=torch.float16),
                "row_norm": torch.ones(rows, dtype=torch.float16),
                "col_norm": torch.ones(columns, dtype=torch.float16),
            }
    return {"layers": layers}


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("layers", [4, 5])
def test_complete_layers_are_balanced_and_map_has_no_ancestor_overlap(family, layers):
    model = tiny_model(family, layers)
    mapping = placement.build_two_gpu_map(model)
    assert mapping["model.embed_tokens"] == "cuda:0"
    assert mapping["model.rotary_emb"] == "cuda:0"
    assert mapping["model.norm"] == mapping["lm_head"] == "cuda:1"
    midpoint = (layers + 1) // 2
    assert [mapping[f"model.layers.{i}"] for i in range(layers)] == ["cuda:0"] * midpoint + ["cuda:1"] * (layers - midpoint)
    assert set(mapping.values()) == {"cuda:0", "cuda:1"}
    assert all(not b.startswith(a + ".") for a in mapping for b in mapping if a != b)
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_eight_gpu_map_assigns_complete_balanced_layer_ranges():
    model = tiny_model(layers=80)
    mapping = placement.build_multi_gpu_map(model, 8)
    assert mapping["model.embed_tokens"] == "cuda:0"
    assert mapping["model.norm"] == mapping["lm_head"] == "cuda:7"
    assignments = [mapping[f"model.layers.{index}"] for index in range(80)]
    for device in range(8):
        assert assignments[device * 10 : (device + 1) * 10] == [f"cuda:{device}"] * 10
    assert all(not child.startswith(parent + ".") for parent in mapping for child in mapping if parent != child)


def test_hybrid_map_balances_resident_layers_and_offloads_tail_to_cpu():
    model = tiny_model(layers=80)
    mapping = placement.build_multi_gpu_map(model, 3, cpu_offload_layers=23)
    assignments = [mapping[f"model.layers.{index}"] for index in range(80)]

    assert assignments[:19] == ["cuda:0"] * 19
    assert assignments[19:38] == ["cuda:1"] * 19
    assert assignments[38:57] == ["cuda:2"] * 19
    assert assignments[57:] == ["cpu"] * 23
    assert mapping["model.embed_tokens"] == "cuda:0"
    assert mapping["model.norm"] == mapping["lm_head"] == "cuda:2"


@pytest.mark.parametrize("kind", ["embedding_head", "across_layers"])
def test_tied_weights_cannot_cross_device_boundaries(kind):
    model = tiny_model()
    if kind == "embedding_head":
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        model.model.layers[3].self_attn.q_proj.weight = model.model.layers[0].self_attn.q_proj.weight
    with pytest.raises(ValueError, match="[Tt]ied"):
        placement.build_two_gpu_map(model)


def test_ties_within_one_device_are_allowed():
    model = tiny_model()
    model.model.layers[1].self_attn.q_proj.weight = model.model.layers[0].self_attn.q_proj.weight
    assert placement.build_two_gpu_map(model)["model.layers.1"] == "cuda:0"


@pytest.mark.parametrize("visible", [0, 1])
def test_split_requires_at_least_two_visible_cuda_devices(monkeypatch, visible):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: visible > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
    with pytest.raises(ValueError, match="at least two"):
        placement.place_evaluation_model(tiny_model(), "cuda:0", split_gpus=True)


def test_split_dispatch_uses_every_visible_cuda_device(monkeypatch):
    model = tiny_model(layers=8)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    def fake_dispatch(actual, **kwargs):
        calls.append(kwargs)
        return actual

    monkeypatch.setattr(placement, "dispatch_model", fake_dispatch)
    monkeypatch.setattr(placement, "_check_resident_placement", lambda *a: None)
    result = placement.place_evaluation_model(model, "cuda:0", split_gpus=True)
    assert set(calls[0]["device_map"].values()) == {"cuda:0", "cuda:1", "cuda:2", "cuda:3"}
    assert result.hf_device_map == calls[0]["device_map"]


def test_split_dispatch_accepts_explicit_cpu_offload_layers(monkeypatch):
    model = tiny_model(layers=8)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    def fake_dispatch(actual, **kwargs):
        calls.append(kwargs)
        return actual

    monkeypatch.setattr(placement, "dispatch_model", fake_dispatch)
    monkeypatch.setattr(placement, "_check_resident_placement", lambda *a: None)
    result = placement.place_evaluation_model(
        model,
        "cuda:0",
        split_gpus=True,
        cpu_offload_layers=2,
    )

    assert list(result.hf_device_map.values()).count("cpu") == 2
    assert calls[0]["main_device"] == "cuda:0"


def test_single_device_path_keeps_original_to_behavior_without_dispatch(monkeypatch):
    model = tiny_model()
    expected = model(torch.tensor([[1, 2, 3]])).logits.detach()
    monkeypatch.setattr(placement, "dispatch_model", lambda *a, **kw: pytest.fail("Unexpected dispatch"))
    actual = placement.place_evaluation_model(model, torch.device("cpu"))
    assert actual is model
    assert not hasattr(model, "hf_device_map")
    torch.testing.assert_close(model(torch.tensor([[1, 2, 3]])).logits, expected)


def test_dispatch_is_resident_and_records_hf_map(monkeypatch):
    model = tiny_model()
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def fake_dispatch(actual, **kwargs):
        assert all(p.device.type == "cpu" for p in actual.parameters())
        calls.append(kwargs)
        return actual

    monkeypatch.setattr(placement, "dispatch_model", fake_dispatch)
    monkeypatch.setattr(placement, "_check_resident_placement", lambda *a: None)
    result = placement.place_evaluation_model(model, "cuda:0", split_gpus=True)
    assert result is model
    assert calls[0]["main_device"] == "cuda:0"
    assert calls[0]["offload_buffers"] is False
    assert "offload_dir" not in calls[0] and "state_dict" not in calls[0]
    assert result.hf_device_map == calls[0]["device_map"]


def test_residency_audit_rejects_cpu_weights_for_cuda_map():
    model = tiny_model()
    with pytest.raises(ValueError, match="resident|placement"):
        placement._check_resident_placement(model, placement.build_two_gpu_map(model))


def test_residency_audit_accepts_cpu_or_meta_storage_for_offloaded_modules():
    model = tiny_model()
    cpu_map = {name: "cpu" for name in placement.build_two_gpu_map(model)}
    placement._check_resident_placement(model, cpu_map)

    model.model.layers[-1].to_empty(device="meta")
    placement._check_resident_placement(model, cpu_map)


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_split_rejects_a_different_primary_input_device(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="first visible"):
        placement.place_evaluation_model(tiny_model(), device, split_gpus=True)


def test_split_rejects_unmaterialized_cpu_load(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="CPU-loaded"):
        placement.place_evaluation_model(tiny_model().to("meta"), "cuda:0", split_gpus=True)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_actual_accelerate_cpu_hooks_use_freshly_reconstructed_weights(family):
    from experiments.remote_runner import evaluate_logical_checkpoint_lm_eval as evaluator
    plain = tiny_model(family, layers=2)
    hooked = tiny_model(family, layers=2)
    hooked.load_state_dict(plain.state_dict())
    # Exercise the real dispatch/hooks implementation on CPU; physical CUDA
    # transfer and two-card residency remain the separate GPU smoke contract.
    cpu_map = {name: "cpu" for name in placement.build_two_gpu_map(hooked)}
    hooked = placement.dispatch_model(hooked, device_map=cpu_map, main_device="cpu", force_hooks=True,
                                     offload_buffers=False,
                                     skip_keys=getattr(hooked, "_skip_keys_device_placement", None))
    checkpoint = checkpoint_for(plain)
    evaluator.reconstruct_checkpoint(plain, checkpoint, torch.device("cpu"))
    evaluator.reconstruct_checkpoint(hooked, checkpoint, torch.device("meta"))
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(hooked(ids).logits, plain(ids).logits, rtol=0, atol=0)


def test_reconstruction_uses_weight_device_and_keeps_unquantized_parameters():
    from experiments.remote_runner import evaluate_logical_checkpoint_lm_eval as evaluator
    from experiments.nowag_d6.logical_state import get_linear_modules
    model = tiny_model(layers=2)
    embedding = model.model.embed_tokens.weight.detach().clone()
    head = model.lm_head.weight.detach().clone()
    # The legacy device argument must not route the reconstruction to meta.
    audit = evaluator.reconstruct_checkpoint(model, checkpoint_for(model), torch.device("meta"))
    assert audit["quantized_linears"] == 14
    for index in range(2):
        for module in get_linear_modules(model, index).values():
            assert module.weight.device.type == "cpu"
            torch.testing.assert_close(module.weight, torch.full_like(module.weight, .25))
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding)
    torch.testing.assert_close(model.lm_head.weight, head)


def test_scoring_targets_follow_logits_device_without_cuda(monkeypatch):
    from experiments.nowag_d6 import v19_validation_score as scorer
    def model(**kwargs):
        return SimpleNamespace(logits=torch.empty(1, 4, 8, device="meta"))
    calls = []

    def fake_cross_entropy(prediction, targets, reduction):
        assert prediction.device == targets.device == torch.device("meta")
        calls.append(targets.numel())
        return torch.tensor(1.)

    monkeypatch.setattr(scorer.F, "cross_entropy", fake_cross_entropy)
    result = scorer.next_token_nll(model, torch.tensor([[1, 2, 3, 4]]), torch.device("cpu"), chunk_size=2)
    assert calls == [2, 1]
    assert result["predicted_tokens"] == 3


def test_split_cli_is_opt_in(monkeypatch):
    from experiments.remote_runner import evaluate_logical_checkpoint_lm_eval as evaluator
    arguments = ["evaluator", "--model", "m", "--checkpoint", "c", "--output-dir", "o"]
    monkeypatch.setattr("sys.argv", arguments)
    assert not evaluator.parse_args().split_gpus
    monkeypatch.setattr("sys.argv", arguments + ["--split-gpus"])
    assert evaluator.parse_args().split_gpus
