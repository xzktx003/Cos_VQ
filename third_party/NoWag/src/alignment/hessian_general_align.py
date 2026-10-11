import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Tuple, Optional, Union, List

# import src.compression_parent as compress_parent
import warnings
import tqdm
import wandb


def loss(reconstructed_weights, original_weights, hessian):
    diff = original_weights - reconstructed_weights
    loss = torch.einsum("ij,jk,ik->", diff, hessian, diff)
    # print("loss", loss)
    # print("hessian", hessian)
    return loss


class dummy_lr_scheduler:
    def __init__(self):
        pass

    def step(self, x):
        pass


def initialize_optimizer(compression_module, lr, lr_multiplier, patience_scheduler):
    params = []
    if isinstance(lr, dict):
        for name, param in compression_module.named_parameters():
            # print(name)
            if param.requires_grad:
                # search for the name or substring in the dictionary
                found = False
                for key in lr.keys():
                    if key in name:
                        params.append({"params": param, "lr": lr[key]})
                        found = True
                        break
                if not found:
                    params.append({"params": param, "lr": lr["default"]})
        optimizer = torch.optim.Adam(params)

    else:
        print("here")
        for name, param in compression_module.named_parameters():
            # print(name)
            if param.requires_grad:
                params.append({"params": param})

        optimizer = torch.optim.Adam(params, lr=lr)
    # print("lr_multiplier", lr_multiplier)
    if lr_multiplier < 1:
        # print("reducing lr on plateau scheduler")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience_scheduler, factor=lr_multiplier
        )
    else:
        lr_scheduler = dummy_lr_scheduler()

    return optimizer, lr_scheduler


@torch.enable_grad()
def align(
    compression_module,
    original_weights: torch.FloatTensor,
    train_hessian: torch.FloatTensor,
    val_hessian: Optional[torch.FloatTensor] = None,
    lr: Union[float, dict[str, float]] = 1e-3,
    optimizer=None,
    lr_multiplier: float = 1,  # decay the lr by this factor every time the val loss increases
    n_iters: int = 100,
    val_every: int = 1,
    discrete_update_every: int = 1,
    reinitialize_optimizer: bool = True,
    clip_grad: float = -1,
    verbose: Union[bool, int] = 10,
    low_bound: float = 1e-5,
    patience: int = 10,
    patience_scheduler: int = 2,
    eps: float = 1e-5,
    discrete_update_kwargs: Optional[dict] = {},
    n_discrete_updates: int = 1,
    log_wandb: bool = False,
):
    """aligns the compression module to the hessian of the training dataset

    Args:
        compression_module (compress_parent.CompressorParent): the compression module we want to aling
        original_weights (torch.FloatTensor): the original weights of the model
        train_hessian (torch.FloatTensor): the hessian of the training dataset
        val_hessian (Optional[torch.FloatTensor], optional): the hessian of the validation dataset, if None, we don't use it. Defaults to None.
        lr (float, optional): the learning rate for the optimizer. Defaults to 1e-3.
        lr_multiplier (float, optional): multiply the learning rate by this factor every time the validation loss increases. Defaults to 1.
        val_every (int, optional): validate the model every this number of iterations on the validation hessian. Defaults to 1.
        discrete_update_every (int, optional): update the discrete variables every this number of iteration. Defaults to 1.
        clip_grad (float, optional): clip the gradient norm to this value. Defaults to -1 in which case we don't clip the gradient.
        verbose (bool, optional): print every this number of iterations. If False, we don't print anything. If True, we print at the end only. Defaults to False.
        low_bound (float, optional): the lower bound for the error, below which we stop training. Defaults to 1e-5.
        patience (int, optional): the patience for the early stop, if the loss has not improved by eps for this number of iterations, we stop training. Defaults to 10.
        patience_scheduler (int, optional): the patience for the learning rate scheduler. Defaults to 2.
        eps (float, optional): the minimum improvement in the loss to consider it as an improvement. Defaults to 1e-5.
    Returns:
        compress_parent.CompressorParent: the aligned compression module
    """

    # initialize the optimizer
    # for name, param in compression_module.named_parameters():
    #     print(name, param.requires_grad, param.shape, param.numel())
    optimizer, lr_scheduler = initialize_optimizer(
        compression_module, lr, lr_multiplier, patience_scheduler
    )

    # initialize the best loss
    best_loss = float("inf")
    best_state_dict = copy.deepcopy(compression_module.state_dict())

    patience_counter = 0
    val_loss = None
    print("discrete_update_every", discrete_update_every)
    # print("val_hessian", val_hessian)
    for i in range(n_iters):
        optimizer.zero_grad()
        reconstructed_weights = compression_module.reconstruct()
        train_loss = loss(reconstructed_weights, original_weights, train_hessian)
        # print(train_loss)
        if i % val_every == 0 and val_hessian is not None:
            with torch.no_grad():
                val_reconstructed_weights = compression_module.reconstruct()
                val_loss = loss(
                    val_reconstructed_weights, original_weights, val_hessian
                ).item()

            if val_loss < best_loss - eps and val_loss > low_bound:
                best_loss = val_loss
                best_state_dict = copy.deepcopy(compression_module.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter == patience:
                    break

            if val_loss < low_bound:
                print("early stopping low bound", low_bound, "val_loss", val_loss)
                break

        if val_hessian is None:
            if train_loss < best_loss - eps and train_loss > low_bound:
                patience_counter = 0
            else:
                patience_counter += 1
                # print("increasing patience counter", patience_counter, "patience", patience)
                if patience_counter >= patience:
                    print("ran out of patience, patience used", patience)
                    break
            if train_loss < best_loss and train_loss > low_bound:
                best_loss = train_loss.item()
                best_state_dict = copy.deepcopy(compression_module.state_dict())
                best_state_dict_idx = i
            if train_loss < low_bound:
                print("early stopping low bound", low_bound, "train_loss", train_loss)
                break

        train_loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(compression_module.parameters(), clip_grad)

        # for name, param in compression_module.named_parameters():
        #     print(name, torch.any(~torch.isfinite(param.grad)))
        optimizer.step()
        if val_loss is not None:
            lr_scheduler.step(val_loss)
        else:
            # print("no val loss")
            lr_scheduler.step(train_loss)

        if i % discrete_update_every == 0 and i != 0 and discrete_update_every > 0:
            # with torch.no_grad():
            #     optimizer.zero_grad()
            #     reconstructed_weights = compression_module.reconstruct()
            #     loss_pre_discrete = loss(reconstructed_weights, original_weights, train_hessian)
            #     # print("loss before discrete update", train_loss)
            # load the best state dict
            # if (best_state_dict_idx - 1)%discrete_update_every == 0:
            #     print("checking")
            #     #check that it is the same as the post discrete state dict
            #     for key in post_discrete_state_dict.keys():
            #         assert torch.all(post_discrete_state_dict[key] == best_state_dict[key]), f"key {key} post_discrete_state_dict[key] {post_discrete_state_dict[key]} best_state_dict[key] {best_state_dict[key]}"

            compression_module.load_state_dict(best_state_dict)
            print("loading best state dict")
            print(
                "loss before discrete update",
                compression_module.get_reconstruction_error().item(),
            )
            for i in range(n_discrete_updates):
                n_updated = compression_module.update_discrete(**discrete_update_kwargs)
                if n_updated == 0:
                    print("no discrete variables updated, breaking")
                    break
            # otherwise, we have to continue until we get a non-zero update
            with torch.no_grad():
                post_discrete_loss = loss(
                    reconstructed_weights,
                    original_weights,
                    train_hessian if val_hessian is None else val_hessian,
                )
                if post_discrete_loss < best_loss:
                    best_loss = post_discrete_loss
                    best_state_dict = copy.deepcopy(compression_module.state_dict())
                    best_state_dict_idx = i
                    print("updating best state dict")
                print("loss after discrete update", post_discrete_loss)
            # raise NotImplementedError

            # if n_updated == 0:
            #     print("no discrete variables updated, breaking")
            #     break
            # optimizer.zero_grad()
            if reinitialize_optimizer:
                optimizer, lr_scheduler = initialize_optimizer(
                    compression_module, lr, lr_multiplier, patience_scheduler
                )
            # with torch.no_grad():
            #     reconstructed_weights = compression_module.reconstruct()
            #     loss_post_discrete = loss(reconstructed_weights, original_weights, train_hessian)

            # assert loss_post_discrete < loss_pre_discrete, f"loss_post_discrete {loss_post_discrete} loss_pre_discrete {loss_pre_discrete}"

            # raise NotImplementedError

        if verbose and i % verbose == 0:
            print(
                f'iter {i}/{n_iters}, train loss {train_loss.item()}, val loss {val_loss}, lr {round(optimizer.param_groups[0]["lr"], 6)} best_state_dict_idx {best_state_dict_idx}'
            )
        if log_wandb:
            wandb.log(
                {"train_loss": train_loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            )

    compression_module.load_state_dict(best_state_dict)
    print("hessian best loss", best_loss)
    return compression_module, best_loss
