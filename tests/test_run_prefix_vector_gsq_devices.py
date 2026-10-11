import sys
import types

import torch
from torch import nn

if "lion_pytorch" not in sys.modules:
    sys.modules["lion_pytorch"] = types.SimpleNamespace(Lion=object)

from experiments.nowag_d6.run_prefix_vector_gsq import prepare_inputs_for_experiment


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_prepare_inputs_routes_token_ids_to_model_input_device() -> None:
    model = _DummyModel()
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    routed = prepare_inputs_for_experiment(
        model,
        layer_index=0,
        inputs=token_ids,
        inputs_are_hidden=False,
    )
    assert routed.device == model.model.embed_tokens.weight.device


def test_prepare_inputs_routes_hidden_states_to_layer_device() -> None:
    model = _DummyModel()
    hidden = torch.randn(1, 3, 4)
    routed = prepare_inputs_for_experiment(
        model,
        layer_index=0,
        inputs=hidden,
        inputs_are_hidden=True,
    )
    assert routed.device == model.model.layers[0].weight.device
