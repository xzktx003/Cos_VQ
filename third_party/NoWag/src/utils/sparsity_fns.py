# a collection of sparse selection functions
import torch
import torch.nn as nn
import torch.nn.functional as F

# each function should have 3 arguments: weight, hessian, and sparsity


def magnitude_unstructured_sparsity(
    weight: torch.FloatTensor, hessian: torch.FloatTensor, sparsity: float
):
    """Magnitude based unstructured sparsity"""
    threshold = torch.kthvalue(
        torch.abs(weight).view(-1), int(sparsity * weight.numel())
    ).values
    return torch.abs(weight) < threshold


def magnitude2_row_column_sparsity(
    weight: torch.FloatTensor, hessian: torch.FloatTensor, sparsity: float
):
    """Magnitude based row-column sparsity"""
    norm_0 = torch.norm(weight, p=2, dim=0)
    norm_1 = torch.norm(weight, p=2, dim=1)

    threshold = torch.kthvalue(
        torch.concat([norm_0, norm_1]), int(sparsity * weight.numel())
    ).values

    return (norm_0 < threshold).unsqueeze(0) & (norm_1 < threshold).unsqueeze(1)
