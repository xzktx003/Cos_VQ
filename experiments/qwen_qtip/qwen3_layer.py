"""Qwen3 decoder-layer surface used by QTIP's LLaMA-oriented layer runner."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Qwen3QTIPDecoderLayer(nn.Module):
    """Preserve Qwen3 semantics when a decoder layer is invoked in isolation.

    Hugging Face Qwen3 normally computes RoPE once in ``Qwen3Model.forward``
    and passes ``position_embeddings`` to every decoder layer.  QTIP's
    layerwise Hessian and fine-tuning code invokes a decoder layer directly.
    This adapter owns the same rotary module and supplies the missing shared
    embedding without changing attention, MLP, norm, or Linear parameters.
    """

    def __init__(self, layer: nn.Module, rotary_emb: nn.Module) -> None:
        super().__init__()
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.rotary_emb = rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        if cache_position is None:
            cache_position = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        return outputs
