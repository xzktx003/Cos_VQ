import os
import time

import torch
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Literal

# from vector_quantizer import *
import tqdm

# from quant import *
import random
import numpy as np

import src.compression_parent as compression_parent
import src.quantize_compress as qc

from src.utils.model_utils import find_layers, get_model, inference_layer
import src.data as data
import src.utils.utils as utils
import yaml
import transformers.models.llama.modeling_llama as llama

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


class args_load:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            self.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))

    @staticmethod
    def load(yaml_path: str):
        return args_load(yaml_path)


def load_layer_from_checkpoint(
    checkpoints: dict[str:str],
    layer: nn.Module,
    layer_idx: int,
    add_bias: bool = False,
    base_model: str = "llama",
    key_no_exist_handling: Literal["raise", "ignore", "warn"] = "raise",
    quantizer_type: str = "",
    clean: bool = True,
    device: str = "cpu",
    cache_reconstruct: bool = False,
    load_checkpoints: bool = True,
):

    sublayer_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ]
    n_bits, n_params = 0, 0

    for name in tqdm.tqdm(sublayer_names, leave=True, disable=True):
        parent_module = getattr(layer, name.split(".")[0])
        module = getattr(parent_module, name.split(".")[1])
        # if device is None:
        original_device = next(module.parameters()).device

        original_dtype = next(iter(module.parameters())).dtype

        if device is not None:
            # device = original_device
            module.to(device)

        sublayer_full_name = f"{base_model}/layer_{layer_idx}/{name}"
        if sublayer_full_name not in checkpoints:
            sublayer_full_name = (
                f"layer_{layer_idx}/{name}"  # try an abbreviated version I used before
            )
            if sublayer_full_name not in checkpoints:
                if key_no_exist_handling == "raise":
                    raise ValueError(
                        f"Checkpoint for {sublayer_full_name} not found in keys: {checkpoints.keys()}"
                    )
                elif key_no_exist_handling == "ignore":
                    continue
                elif key_no_exist_handling == "warn":
                    print(
                        f"Checkpoint for {sublayer_full_name} not found in keys: {checkpoints.keys()}"
                    )
                    continue
                # raise ValueError(f"Checkpoint for {sublayer_full_name} not found in keys: {checkpoints.keys()}")
        checkpoint_path = checkpoints[sublayer_full_name]
        checkpoint_args = yaml.load(
            open(checkpoint_path.replace(".pt", "_args.yaml"), "r"),
            Loader=yaml.FullLoader,
        )
        compression_type = checkpoint_args["compression_type"]

        if not hasattr(module, "bias"):
            module.bias = None
        if not hasattr(module, "weight"):
            module.weight = module.reconstruct()

        print(compression_type)
        if "LinearVQ" in compression_type or "quantize" in compression_type:
            print("Loading VQ compressed layer")
            # 确保使用LinearVQ层类型
            if not isinstance(module, qc.LinearVQ):
                new_layer = qc.LinearVQ(module.weight, module.bias, add_bias)
                new_layer.blank_recreate(**checkpoint_args["quantizer_kwargs"])
                module = new_layer
            else:
                # 如果已经是LinearVQ类型，直接加载参数
                module.blank_recreate(**checkpoint_args["quantizer_kwargs"])

        if clean:
            module.clean()

        if load_checkpoints:

            try:
                state_dict = torch.load(
                    checkpoint_path,
                    weights_only=False,
                    map_location=(
                        torch.device(device) if device is not None else original_device
                    ),
                )
            except RuntimeError:
                state_dict = torch.load(checkpoint_path, weights_only=False)

            if "quantize" in compression_type:
                state_dict = {
                    s.replace("quantizer.", ""): v
                    for s, v in state_dict.items()
                    if "reference" not in s
                }
                # rename "assignments" to "codes"
                state_dict = {
                    s.replace("codes", "assignments"): v for s, v in state_dict.items()
                }

                # state_dict["normalizer.original_shape"] = torch.tensor([new_layer.out_features, new_layer.in_features], dtype=torch.int32)
                # print(state_dict["normalizer.norms.0"])
                state_dict["normalizer.norms.0"] = state_dict["normalizer.norms.0"][
                    : module.in_features
                ]

            module.load_state_dict(state_dict, strict=True)

            if cache_reconstruct:
                module.cache_reconstruct()

        module.to(original_dtype)
        module.to(original_device)
        # print("new_layer.quantization_compressor.
        delattr(parent_module, name.split(".")[1])
        setattr(parent_module, name.split(".")[1], module)
        utils.clean()
        n_bits += module.get_n_bits()
        n_params += module.get_n_original_parameters()
        # utils.get_gpu_memory(device)
    utils.clean()
    return layer, n_bits, n_params


@torch.no_grad()
def load_model_from_checkpoints(
    checkpoints: dict[str:str],
    base_model: str,
    model: Optional[llama.LlamaForCausalLM] = None,
    add_bias: Optional[bool] = False,
    key_no_exist_handling: Literal["raise", "ignore", "warn"] = "raise",
    disable_tqdm: Optional[bool] = False,
    log_wandb: Optional[bool] = False,
    quantizer_type: Optional[str] = "",
    clean: Optional[bool] = True,
    device: Optional[str] = "cpu",
    cache_reconstruct: Optional[bool] = False,
    load_checkpoints: Optional[bool] = True,
) -> Tuple[llama.LlamaForCausalLM, float, int]:
    """
    Load a model from a checkpoint of each individual layer.

    Args:
        checkpoints (dict[str:str]): A dictionary of the form {layer_name: checkpoint_path}
        model (llama.LlamaForCausalLM): The model to load the checkpoints into.
        add_bias (bool, optional): Whether to add a bias to the model. Defaults to False. turn to true for

    """
    print("disable_tqdm", disable_tqdm)
    print("load_checkpoints", load_checkpoints)
    # args = args_load(os.path.join(checkpoint_path, "args.yaml"))
    # if not hasattr(args,"compression_type"):
    #     #assume that the model is quantized
    #     args.compression_type = "quantized"

    n_bits = 0
    n_params = 0

    if model is None:
        model = get_model(base_model)
        model.to("cpu")

    layers = model.model.layers
    # original_dtype = next(iter(model.parameters())).dtype

    sublayer_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ]

    for i in tqdm.tqdm(
        range(len(layers)), desc="Loading checkpoints", disable=disable_tqdm
    ):
        layer = layers[i]
        layer, layer_n_bits, layer_n_params = load_layer_from_checkpoint(
            checkpoints,
            layer,
            i,
            add_bias,
            base_model,
            key_no_exist_handling,
            quantizer_type=quantizer_type,
            clean=clean,
            device=device,
            cache_reconstruct=cache_reconstruct,
            load_checkpoints=load_checkpoints,
        )
        n_bits += layer_n_bits
        n_params += layer_n_params
        layers[i] = layer

    # print("bpv", n_bits / n_params)
    # if log_wandb:
    #     wandb.log({f"{base_model}/bpv": n_bits / n_params})
    # model.to(original_dtype)

    return model, n_bits, n_params
