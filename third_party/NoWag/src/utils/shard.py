"""Simple sharded model from QIUP # caoebook"""

# import glog
import torch
from torch import nn
import tqdm
import time


def get_graph_wrapper(cls, device=0):

    class GraphWrapper(cls):

        def __init__(self, *args, **kwargs):
            super(GraphWrapper, self).__init__(*args, **kwargs)
            self.built_graph = False
            self.graph_device = device

        def forward(self, *args, **kwargs):
            with torch.cuda.device(self.graph_device):
                if not self.built_graph:
                    self.static_args = args
                    self.static_kwargs = kwargs

                    s = torch.cuda.Stream(device=self.graph_device)
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        super(GraphWrapper, self).forward(
                            *self.static_args, **self.static_kwargs
                        )
                    torch.cuda.current_stream().wait_stream(s)

                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=s):
                        self.static_output = super(GraphWrapper, self).forward(
                            *self.static_args, **self.static_kwargs
                        )

                    self.built_graph = True
                    print("Built CUDA graph of model.")

                # these two loops take < 1e-4 seconds for llama2
                for i in range(len(args)):
                    if isinstance(args[i], torch.Tensor):
                        self.static_args[i].copy_(args[i])
                for kw in kwargs:
                    if isinstance(kwargs[kw], torch.Tensor):
                        self.static_kwargs[kw].copy_(kwargs[kw])

                self.graph.replay()
                return self.static_output

        def reset(self):
            if self.built_graph:
                del self.static_args, self.static_kwargs
                self.built_graph = False

    return GraphWrapper


def convert_args(args, kwargs, device, dtype):

    def convert_tensor(tensor):
        if tensor.dtype == torch.float16 or tensor.dtype == torch.float32:
            tensor = tensor.to(dtype)
        return tensor.to("cpu").to(device)

    dev_args = []
    for i in range(len(args)):
        dev_args.append(
            convert_tensor(args[i]) if isinstance(args[i], torch.Tensor) else args[i]
        )
    for i in kwargs:
        if isinstance(kwargs[i], torch.Tensor):
            kwargs[i] = convert_tensor(kwargs[i])
    return dev_args, kwargs


class Shard(nn.Module):

    def __init__(self, layers, arg_fn):
        super().__init__()
        self.layers = layers
        self.arg_fn = arg_fn

    def forward(self, *args, **kwargs):
        tqdm.tqdm.write(f"input of shard layer: {args[0]}")
        for layer in self.layers:
            output = layer(*args, **kwargs)
            args, kwargs = self.arg_fn(output, args, kwargs)
        tqdm.tqdm.write("output of shard layer: " + str(output))
        return args, kwargs


class ShardTransformer(nn.Module):

    def __init__(self, shards, output_layer, grad_ckpt, train_mode, to_float=True):
        super().__init__()

        # shards is list of [(device, arg_fn, modulelist)]

        self.shards = nn.ModuleList([_["shard"] for _ in shards])
        self.devices = [_["device"] for _ in shards]

        from src.linear_compress import LinearQuantized

        for name, module in self.shards.named_modules():
            if isinstance(module, LinearQuantized):
                module.grad_ckpt = grad_ckpt
                module.change_otf_denormalize(True)
                # module.train_mode = train_mode
        for i in range(len(shards)):
            device = self.devices[i]
            if to_float:
                self.shards[i].float()
            # self.shards[i] = graph_wrapper.get_graph_wrapper(Shard, device)(self.shards[i], shards[i]['arg_fn']).to(device)
            self.shards[i] = Shard(self.shards[i], shards[i]["arg_fn"]).to(device)
        self.dtype = torch.float32 if to_float else torch.float16
        self.output_layer = output_layer["layer"].to(0)
        self.output_layer_fn = output_layer["fn"]
        self.grad_ckpt = grad_ckpt

    def manifest(self, *args, **kwargs):
        for i in range(len(self.shards)):
            device = self.devices[i]
            # glog.info(f'manifesting layers on gpu {device}')
            args, kwargs = convert_args(args, kwargs, device, self.dtype)
            self.shards[i](*args, **kwargs)

    def shard_wrapper(self, input):
        i, args, kwargs = input
        return self.shards[i](*args, **kwargs)

    def ckpt_shard(self, i, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            self.shard_wrapper, (i, args, kwargs), use_reentrant=False
        )

    def forward(self, *args, **kwargs):
        tqdm.tqdm.write(f"n_shards: {len(self.shards)}")
        for i in range(len(self.shards)):
            device = self.devices[i]
            args, kwargs = convert_args(args, kwargs, device, self.dtype)
            if self.grad_ckpt:
                args, kwargs = self.ckpt_shard(i, *args, **kwargs)
            else:
                args, kwargs = self.shards[i](*args, **kwargs)
        tqdm.tqdm.write(
            f"before output layer {self.output_layer_fn(args, kwargs).to(0)}"
        )
        return self.output_layer(self.output_layer_fn(args, kwargs).to(0))
