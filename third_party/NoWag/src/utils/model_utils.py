import torch
import torch.nn as nn
import tqdm
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


DEV = torch.device("cuda:0")


def get_model(
    model: str,
    model_path: Optional[str] = None,
    device_map: Optional[str] = None,
    dtype: Optional[str] = None,
) -> Any:
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    import transformers
    model_type = "auto"
    if "llama" in model.lower():
        model_type = "llama"
    elif "qwen" in model.lower():
        model_type = "qwen"

    if model_type == "auto":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model if model_path is None else model_path,
            torch_dtype="auto" if dtype is None else dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model if model_path is None else model_path,
            torch_dtype="auto" if dtype is None else dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map=device_map,
            trust_remote_code=True
        )
    # model.seqlen = 8192
    print("Model loaded.", model)
    return model


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


@torch.no_grad()
def inference_layer(
    layer: nn.Module,
    inps: torch.FloatTensor,
    outs: torch.FloatTensor,
    layer_kwargs: dict = {},
    dev: str = "cuda:0",
    offload_activations: bool = False,
    batch_size: int = 8,
    disable_tqdm: bool = False,
    inplace: bool = True,
) -> torch.FloatTensor:
    """Inference a single layer"""
    # layer.to(dev)
    if offload_activations:
        inps = inps.cpu()
        outs = outs.cpu()
    else:
        inps = inps.to(dev)
        outs = outs.to(dev)

    # tqdm.tqdm.write("inps device: "+str(inps.device))
    for j in tqdm.tqdm(
        range(0, len(inps), batch_size),
        desc="Inference",
        miniters=len(inps) // 100,
        disable=disable_tqdm,
    ):
        # print(j, j+batch_size)
        if offload_activations:
            if inplace:
                inps[j : j + batch_size] = layer(
                    inps[j : j + batch_size].to(dev), **layer_kwargs
                )[0].cpu()
            else:
                outs[j : j + batch_size] = layer(
                    inps[j : j + batch_size].to(dev), **layer_kwargs
                )[0].cpu()
        else:
            if inplace:
                inps[j : j + batch_size] = layer(
                    inps[j : j + batch_size], **layer_kwargs
                )[0]
            else:
                outs[j : j + batch_size] = layer(
                    inps[j : j + batch_size], **layer_kwargs
                )[0]
    # layer.cpu()
    if inplace:
        return inps
    return outs


# @torch.no_grad()
# def get_inps(
#     model,
#     data: Sequence,
#     model_seqlen: int,
#     devices: Sequence[torch.device],
#     offload_activations: bool,
# ) -> Tuple[Sequence[torch.Tensor], Dict]:
#     """
#     mocks model launch to collect inputs to the first model layer
#     :returns: a list of torch tensors with activations for each device in args.devices.
#     Each tensor has shape [nsample_per_device, seq_len, hid_size]
#     """
#     print("catching layer inputs from data", flush=True)
#     layers = model.model.layers
#     device = devices[0] if not offload_activations else torch.device("cpu")

#     if isinstance(data, torch.Tensor) and data.shape[0] == 1:  # given a single long tensor, split it into sequences
#         assert data.ndim == 2, "data must be either a single tensor with a long sequence or a list of pre-cut sequences"
#         num_sequences, num_tokens_dropped = data.numel() // model_seqlen, data.numel() % model_seqlen
#         data = [data[:, i * model_seqlen : (i + 1) * model_seqlen].to(device) for i in range(num_sequences)]
#         print(f"Got {len(data)} sequences of {model_seqlen} tokens, dropped last {num_tokens_dropped} tokens")
#         del num_sequences, num_tokens_dropped

#     assert all(sequence.shape[1] == model_seqlen for sequence in data)
#     model.to(devices[0])
#     emb = model.get_input_embeddings()
#     emb_device = emb.weight.device
#     if emb_device.type != "cuda":
#         emb = emb.to(device)
#         # opt has other embeddings
#         if model.config.model_type == "opt":
#             model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
#             if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
#                 model.model.decoder.project_in = model.model.decoder.project_in.to(device)
#     device = emb.weight.device  # now default device is the one where the embeddings are.
#     layer_device = next(layers[0].parameters()).device
#     layers[0] = layers[0].to(device)

#     dtype = next(iter(model.parameters())).dtype
#     nsamples_per_device = (len(data) - 1) // len(devices) + 1
#     inps = [
#         torch.zeros(
#             (min(nsamples_per_device, len(data) - i * nsamples_per_device), model_seqlen, model.config.hidden_size),
#             dtype=dtype,
#             device=devices[i] if not offload_activations else "cpu",
#             pin_memory=offload_activations,
#         )
#         for i in range(len(devices))
#     ]
#     forward_arg_names = ["attention_mask", "position_ids"]

#     cache = {"i": 0, "alibi": None}

#     class CatcherExit(Exception):
#         pass

#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module

#         def forward(self, inp, **kwargs):
#             inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
#             cache["i"] += 1
#             for forward_arg_name in forward_arg_names:
#                 cache[forward_arg_name] = kwargs.get(forward_arg_name)
#             raise CatcherExit()

#     layers[0] = Catcher(layers[0])
#     saved_num_threads = torch.get_num_threads()
#     torch.set_num_threads(min(16, saved_num_threads))
#     for batch_inps in data:
#         try:
#             if isinstance(batch_inps, (list, tuple)):
#                 batch_inps, *_ = batch_inps
#             batch_inps = batch_inps.to(device)
#             # call model.forward to trigger the Catcher
#             model(batch_inps, attention_mask=torch.ones_like(batch_inps))
#         except CatcherExit:
#             pass  # exit after catcher finished without running the rest of the model layers

#     torch.set_num_threads(saved_num_threads)
#     layers[0] = layers[0].module

#     layers[0] = layers[0].to(layer_device)
#     model.get_input_embeddings().to(emb_device)
#     torch.cuda.empty_cache()

#     forward_kwargs = {k: cache[k] for k in forward_arg_names}
#     assert cache["i"] == sum(len(inp_tensor) for inp_tensor in inps), "internal error: found empty rows in inps"
#     model.to("cpu")
#     return inps, forward_kwargs
