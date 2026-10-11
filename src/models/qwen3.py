import os
import time
import torch
import math
import gc
import torch.distributed as dist
from contextlib import ExitStack
from safetensors.torch import save_file as safe_save_file
from accelerate.utils import set_module_tensor_to_device
from .base import BaseModelWrapper
from src.quantization import is_nowag_state
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_ppl_layer

class Qwen3Wrapper(BaseModelWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)
        self.layer_prefix = "model.layers"
        self.num_layers = len(self.model.model.layers)
    
    def get_mlp_input(self, batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        additional_layer_inputs = self._build_layer_inputs(batch.shape[0])
        if "position_embeddings" not in additional_layer_inputs:
            position_ids = torch.arange(
                batch.shape[1], device=batch.device, dtype=torch.long
            ).unsqueeze(0).expand(batch.shape[0], -1)
            additional_layer_inputs["position_embeddings"] = self.model.model.rotary_emb(
                batch, position_ids
            )

        hidden_states = current_layer.input_layernorm(batch)
        hidden_states, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
        return hidden_states + batch
    
    def get_mlp_output(self, mlp_input_batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        hidden_states = current_layer.post_attention_layernorm(mlp_input_batch)
        hidden_states = current_layer.mlp(hidden_states)
        return hidden_states + mlp_input_batch

    def _get_mlp_output_with_quantized(self, batch, quantized_weights):
        """Run only the MLP while preserving gradients to replacement weights."""
        current_layer = self.get_layer_module(self.current_layer_idx)
        hooks = []

        class LinearWeightHook:
            def __init__(self, module, quantized_weight):
                self.module = module
                self.quantized_weight = quantized_weight
                self.original_forward = module.forward

            def __enter__(self):
                def new_forward(module_self, x):
                    return torch.nn.functional.linear(
                        x, self.quantized_weight, self.module.bias
                    )

                self.module.forward = new_forward.__get__(
                    self.module, torch.nn.Linear
                )

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.module.forward = self.original_forward

        try:
            for name, module in current_layer.mlp.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                key = f"{self.get_current_layer()}.mlp.{name}"
                if key in quantized_weights:
                    hooks.append(LinearWeightHook(module, quantized_weights[key]))

            with ExitStack() as stack:
                for hook in hooks:
                    stack.enter_context(hook)
                return self._tensor_output(self.get_mlp_output(batch))
        finally:
            hooks.clear()

    def calculate_mse(
        self,
        batch,
        quantized_weights,
        self_attn,
        validation=False,
        accumulation_steps=1,
        defer_item=False,
    ):
        if self_attn or not getattr(self, "dense_mlp_inputs_precomputed", False):
            return super().calculate_mse(
                batch,
                quantized_weights,
                self_attn,
                validation=validation,
                accumulation_steps=accumulation_steps,
                defer_item=defer_item,
            )

        with torch.no_grad():
            out_fp = self._tensor_output(self.get_mlp_output(batch))
        with torch.set_grad_enabled(not validation):
            out_q = self._get_mlp_output_with_quantized(batch, quantized_weights)
            mse = self.loss_fn(out_q, out_fp)
            if not validation:
                (mse / accumulation_steps).backward()
        return mse.detach() if defer_item else mse.item()

    def get_layer_module(self, idx):
        return self.model.model.layers[idx]
    
    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        non_mlp = [
            f"{base}.input_layernorm",
            f"{base}.self_attn",
            f"{base}.post_attention_layernorm"
        ]
        mlp = [
            f"{base}.mlp"
        ]
        return {"non_mlp": non_mlp, "mlp": mlp}

    def move_embed_to(self, device):
        names = self._names_from_ckpt(["model.embed_tokens"])
        if device == "cuda":
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def move_output_heads_to(self, device):
        norm_names = self._names_from_ckpt("model.norm")
        head_names = self._names_from_ckpt("lm_head")
        if device == "cuda":
            self._set_tensors(norm_names + head_names)
            if not head_names and self.model.config.tie_word_embeddings:
                embedding = self.model.model.embed_tokens.weight
                if embedding.device.type == "meta":
                    raise RuntimeError(
                        "tied lm_head requires materialized embedding weights"
                    )
                set_module_tensor_to_device(
                    self.model,
                    "lm_head.weight",
                    self.device,
                    value=embedding.detach(),
                    dtype=embedding.dtype,
                )
        else:
            self._offload_names_to_meta(norm_names + head_names)
            if not head_names and self.model.config.tie_word_embeddings:
                set_module_tensor_to_device(self.model, "lm_head.weight", "meta")

    def save_attention_to_disc(self, pfx):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        compressed_attn_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        non_compressed_attn_layers = ['k_norm', 'q_norm']
        self.quantization_config.config_groups.group_0.weights.group_size = self.groupsize
        for name in compressed_attn_layers:
            base = f"{pfx}.{name}"
            module = self.model.get_submodule(base)
            state = self.temp_weights.get(f"{base}.quant_state")
            if is_nowag_state(state):
                to_save[base + ".weight"] = state["weight"].detach().cpu()
                for field in (
                    "assignments", "scale0", "scale", "log_scale", "codebook",
                    "row_norm", "col_norm", "groupsize",
                ):
                    to_save[base + f".nowag_{field}"] = state[field].detach().cpu()
                continue
            if isinstance(self.temp_weights[f"{base}.scale"], tuple):
                self.temp_weights[f"{base}.scale"] = self.temp_weights[f"{base}.scale"][0]
            compressed = self._compress_weight(
                module.weight.data.detach().cpu(),
                self.temp_weights[f"{base}.scale"].detach().cpu(),
            )
            to_save[base + ".weight_packed"] = compressed["weight_packed"]
            to_save[base + ".weight_scale"]  = self.temp_weights[f"{base}.scale"].detach().cpu()
            to_save[base + ".weight_shape"]  = compressed["weight_shape"]
        for name in non_compressed_attn_layers:
            base = f"{pfx}.{name}"
            module = self.model.get_submodule(base)
            to_save[base + ".weight"] = module.weight.data.detach().cpu()
        safe = pfx.replace('.', '_')
        path = os.path.join(self.save_dir, f"{safe}.safetensors")
        safe_save_file(to_save, path)

    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("wikitext2", self.tokenizer)
        testloader = prepare_test_dataloader(
                dataset=dataset["test"], 
                tokenizer=self.tokenizer, 
                seqlen=self.model.seqlen,
                batch_size=8,
                world_size=self.world_size,
                rank=self.rank
        )
        
        pad_token_id = self.model.config.pad_token_id

        if pad_token_id is not None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()

        self.move_embed_to(self.device)
        self.move_output_heads_to(self.device)

        input_ids_cpu_list = []
        activations_cpu_list = []

        for batch in testloader:
            ids_cpu = batch["input_ids"].to("cpu", non_blocking=True).pin_memory()
            input_ids_cpu_list.append(ids_cpu)
            activations_cpu_list.append(None)
            del batch

        num_batches = len(input_ids_cpu_list)

        additional_layer_inputs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            additional_layer_inputs[k] = v

        idx_copy = self.current_layer_idx
        ppl_start = time.time()

        for i in range(self.num_layers):
            if self.rank == 0:
                report_ppl_layer(i, self.num_layers, elapsed=time.time() - ppl_start)
            self.current_layer_idx = i
            layer_name = f"{self.layer_prefix}.{i}"
            if i <= read_from_disk:
                self.load_from_disc(layer_name)
            else:
                self.move_layer_to_gpu(layer_name)
            layer = self.get_layer_module(i)

            for b in range(num_batches):
                ids_cpu = input_ids_cpu_list[b]

                x_cpu = activations_cpu_list[b]
                if x_cpu is None:
                    ids = ids_cpu.to(self.device, non_blocking=True)
                    x = self.model.model.embed_tokens(ids)
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                x = self.get_mlp_input(x)
                x = self.get_mlp_output(x)

                activations_cpu_list[b] = x.to("cpu", non_blocking=True).pin_memory()

            self.offload_to_meta(layer_name)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        nll_sum = torch.tensor(0.0, device=self.device)
        tok_cnt = torch.tensor(0.0, device=self.device)

        for b in range(num_batches):
            ids_cpu = input_ids_cpu_list[b]
            x_cpu = activations_cpu_list[b]

            input_ids = ids_cpu.to(self.device, non_blocking=True)
            x = x_cpu.to(self.device, non_blocking=True)

            x = self.model.model.norm(x)
            logits = self.model.lm_head(x)

            logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()
            mask = shift_labels != loss_fn.ignore_index
            nll = (nll * mask).sum(dim=1)
            tok = mask.sum(dim=1)

            nll_sum += nll.sum()
            tok_cnt += tok.sum()

        self.move_embed_to("meta")
        self.move_output_heads_to("meta")

        self.current_layer_idx = idx_copy

        if self.world_size > 1:
            dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(tok_cnt, op=dist.ReduceOp.SUM)

        mean_nll = (nll_sum / tok_cnt).item()
        ppl = math.exp(mean_nll)

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return ppl
