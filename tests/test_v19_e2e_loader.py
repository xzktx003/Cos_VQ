import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils

from experiments.nowag_d6.finetune_hard_codebook_e2e import load_e2e_base_model, training_data_description
from experiments.qwen_qtip.qtip_runtime import install_tp_plan_compat


def test_fineweb_training_is_not_reported_as_rp1t():
    result = training_data_description('FineWeb-Edu, first 2048 cached rows')
    assert result['dataset'] == 'FineWeb-Edu, first 2048 cached rows'
    assert 'rp1t' not in result['objective']
    assert 'rp1t' not in result['selection']


def test_e2e_loader_repairs_torch24_tp_registry_for_real_local_model(tmp_path, monkeypatch):
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    install_tp_plan_compat(config)
    config.save_pretrained(tmp_path)
    save_file(dict(LlamaForCausalLM(config).state_dict()), tmp_path / 'model.safetensors')
    monkeypatch.setattr(modeling_utils, 'ALL_PARALLEL_STYLES', None)
    model = load_e2e_base_model(tmp_path)
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    with torch.no_grad():
        assert torch.isfinite(model(torch.tensor([[1, 2, 3]])).logits).all()
