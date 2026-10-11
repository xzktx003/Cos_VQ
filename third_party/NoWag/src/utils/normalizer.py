import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.jit as jit
import os
import sys
from typing import Tuple, Optional, Union, List


def round_to_the_nearest(x, codebook):
    return torch.argmin(torch.abs(x.unsqueeze(0) - codebook.unsqueeze(1)), dim=0)


class Normalizer(nn.Module):
    pass


class Normalizer(nn.Module):

    def __init__(
        self,
        norms: List[Union[torch.Tensor | None]],
        zeros: List[Union[torch.Tensor | None]],
        norm_order: List[int] = [0, 1],
        shape: Tuple[int, int] = (0, 0),
    ):
        super(Normalizer, self).__init__()
        self.norm_order = norm_order

        self.norms = nn.ParameterList([nn.Parameter(norm) for norm in norms])
        self.zeros = nn.ParameterList([nn.Parameter(zero) for zero in zeros])

        self.original_shape = shape

    def denormalize(
        self, normalized_weight: torch.FloatTensor, debias=True
    ) -> torch.FloatTensor:

        # print("reversed_norm_order", reversed_norm_order)
        # print("norm_order", self.norm_order)
        # for norm in self.norms:
        #     # print(norm.shape)
        for j in reversed(range(len(self.norm_order))):
            i = self.norm_order[j]
            # print("i",i)
            # print("j",j)
            if self.norms[j] is not None and self.norms[j].numel() > 0:
                # print(self.norms[i].shape)
                # print(normalized_weight.shape[reversed_norm_order[i]])
                # print(self.norms[i][:normalized_weight.shape[reversed_norm_order[i]]].unsqueeze(i).shape)
                normalized_weight = normalized_weight * self.norms[j].unsqueeze(i)
            if self.zeros[j] is not None and self.zeros[j].numel() > 0 and debias:
                normalized_weight = normalized_weight + self.zeros[j].unsqueeze(i)
            # print("j",j,"normalized_weight", normalized_weight)
        # print("========")
        # raise ValueError("not implemented")
        return normalized_weight

    def denormalize_otf_in(
        self, input_activation: torch.FloatTensor
    ) -> torch.FloatTensor:
        for j, i in enumerate(self.norm_order):
            if i == 0:
                input_activation = input_activation * self.norms[j]
        return input_activation

    def denormalize_otf_out(
        self, output_activation: torch.FloatTensor
    ) -> torch.FloatTensor:
        for j, i in enumerate(self.norm_order):
            if i == 1:
                output_activation = output_activation * self.norms[j]
                if self.zeros[j] is not None and self.zeros[j].numel() > 0:
                    raise ValueError("not implemented")

        return output_activation

    def normalize(self, weight: torch.FloatTensor) -> torch.FloatTensor:
        """normalize the input weight matrix"""
        normalized_weight = weight.clone()
        for j, i in enumerate(self.norm_order):
            if self.zeros[j] is not None and self.zeros[j].numel() > 0:
                normalized_weight = normalized_weight - self.zeros[j].unsqueeze(i)
            if self.norms[j] is not None and self.norms[j].numel() > 0:
                normalized_weight = normalized_weight / self.norms[j].unsqueeze(i)

        return normalized_weight

    @staticmethod
    def normalize_init(
        weight: torch.FloatTensor,
        norm_order: list[int] = [],
        zero: list[bool] = [],
        eps: float = 1e-5,
        norm_rescale: float = True,
        powers: float = 1,
        p: float = 2,
        std_norm: bool = False,
    ) -> Tuple[Normalizer, torch.FloatTensor]:

        original_weight = weight.clone()
        norms = [None] * len(norm_order)
        zeros = [None] * len(norm_order)

        n_out, n_in = weight.shape
        # print(zero)
        # print("norm_order", norm_order)
        for i, dim in enumerate(norm_order):
            # print("dim",dim)
            if zero[i]:
                zeros[i] = torch.mean(weight, dim=dim)
                if norm_rescale:
                    weight = weight - zeros[i].unsqueeze(dim)
                # weight = weight - zeros[dim].unsqueeze(dim)
            if not std_norm:
                norms[i] = torch.norm(weight, dim=dim, p=p) ** powers + eps
            else:
                norms[i] = torch.std(weight, dim=dim) + eps
            if norm_rescale:
                # print("i",i,"weight", weight)
                weight = weight / norms[i].unsqueeze(dim)
                # print("norms[i]", norms[i])
            assert torch.all(torch.isfinite(weight))
            assert torch.all(torch.isfinite(norms[dim]))

        normalizer = Normalizer(norms, zeros, norm_order, (n_out, n_in))
        # print("i", i+1, "norms", norms)
        if not norm_rescale:
            weight = normalizer.normalize(weight)

        # print("="*10)
        denormalized_weight = normalizer.denormalize(weight)
        # print("="*10)
        # print("denormalized_weight", denormalized_weight)
        # print("weight", weight)
        # print("max diff:", torch.max(torch.abs(denormalized_weight - original_weight)))
        # assert torch.allclose(original_weight, denormalized_weight), f"original weight and denormalized weight are not the same original weight: {original_weight} denormalized weight: {denormalized_weight}"

        return normalizer, weight

    @staticmethod
    def blank_recreate(
        weight: torch.FloatTensor,
        norm_order: List[int] = [0, 1],
        zero: List[bool] = [False, False],
        **kwargs
    ) -> Normalizer:

        norms = [None] * len(norm_order)
        zeros = [None] * len(norm_order)

        for i, dim in enumerate(norm_order):
            shape = [s for i, s in enumerate(weight.shape) if i != dim]
            if zero[dim]:
                zeros[i] = torch.zeros(shape, dtype=weight.dtype, device=weight.device)
            norms[i] = torch.ones(shape, dtype=weight.dtype, device=weight.device)

        normalizer = Normalizer(norms, zeros, norm_order, weight.shape)
        return normalizer

    def get_n_bits(self):
        n_bits = 0
        for norm in self.norms:
            if norm is not None:
                n_bits += norm.numel() * 16
        for zero in self.zeros:
            if zero is not None:
                n_bits += zero.numel() * 16
        return n_bits
