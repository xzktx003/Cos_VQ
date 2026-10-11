from pathlib import Path

from src.models.base import BaseModelWrapper


class _Wrapper(BaseModelWrapper):
    def get_mlp_input(self, layer_input):
        raise NotImplementedError

    def get_mlp_output(self, mlp_input_batch):
        raise NotImplementedError

    def get_layer_module(self, idx):
        raise NotImplementedError

    def move_embed_to(self, device):
        raise NotImplementedError

    def move_output_heads_to(self, device):
        raise NotImplementedError

    def _layer_prefixes(self, layer_name):
        return {
            "non_mlp": [
                f"{layer_name}.input_layernorm",
                f"{layer_name}.self_attn",
                f"{layer_name}.post_attention_layernorm",
            ],
            "mlp": [f"{layer_name}.mlp"],
        }

    def ppl_evaluation(self, read_from_disk=-1):
        raise NotImplementedError


def _wrapper(tmp_path):
    wrapper = object.__new__(_Wrapper)
    wrapper.save_dir = str(tmp_path)
    return wrapper


def test_attention_phase_is_resumable_only_when_all_non_mlp_files_exist(tmp_path):
    wrapper = _wrapper(tmp_path)
    layer = "model.layers.12"
    expected = [
        "model_layers_12_input_layernorm.safetensors",
        "model_layers_12_self_attn.safetensors",
        "model_layers_12_post_attention_layernorm.safetensors",
    ]

    for filename in expected[:-1]:
        (Path(tmp_path) / filename).touch()
    assert not wrapper.saved_components_available(layer, ("non_mlp",))

    (Path(tmp_path) / expected[-1]).touch()
    assert wrapper.saved_components_available(layer, ("non_mlp",))
    assert not wrapper.saved_components_available(layer, ("mlp",))
