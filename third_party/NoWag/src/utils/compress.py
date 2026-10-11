import torch
import torch.nn as nn
import math


def hessian_mean_logging(module, input):
    """updates the mean of the hessian"""

    input_flattened = input[0].reshape(-1, module.in_features).to(torch.float32)
    n_new_samples = input_flattened.shape[0]
    # print(n_new_samples)
    # multiply the hessian by:
    # print(module.n_samples)
    module.hessian *= module.n_samples / (module.n_samples + n_new_samples)
    # outer product of the flattened input
    # first scale input_flattened
    module.n_samples += n_new_samples
    input_flattened = input_flattened * math.sqrt(2 / (module.n_samples))
    module.hessian += input_flattened.T @ input_flattened


def hessian_ema_logging(module, input):

    input_flattened = input[0].reshape(-1, module.in_features).to(torch.float32)
    n_new_samples = input_flattened.shape[0]

    # rescale the inputs

    input_flattened = input_flattened * math.sqrt(2 / (n_new_samples))

    module.hessian = module.hesian * (
        module.decay
    ) + input_flattened.T @ input_flattened * (1 - module.decay)


def hessianDiag_mean_logging(module, input):
    """updates the mean of the hessian diagonal"""
    if input[0].shape[0] == 0:
        return
    input_flattened = input[0].reshape(-1, module.in_features).to(torch.float32)
    n_new_samples = input_flattened.shape[0]

    # multiply the hessian by:
    module.hessianDiag *= module.n_samples / (module.n_samples + n_new_samples)
    # outer product of the flattened input
    # first scale input_flattened
    module.n_samples += n_new_samples
    input_flattened = input_flattened * math.sqrt(2 / (module.n_samples))
    module.hessianDiag += (input_flattened**2).sum(dim=0)


def hessianDiag_ema_logging(module, input):
    """updates the mean of the hessian diagonal"""
    if input[0].shape[0] == 0:
        return
    input_flattened = input[0].reshape(-1, module.in_features).to(torch.float32)
    n_new_samples = input_flattened.shape[0]

    # multiply the hessian by:
    module.hessianDiag = (
        module.hessianDiag * module.decay
        + (input_flattened**2).mean(dim=0) * (1 - module.decay) * 2
    )
