import torch
import torch.nn as nn
import torch.optim as optim
try:
    import wandb
except ImportError:
    from compat import wandb
import math
import time
import gc
from lion_pytorch import Lion
import torch.nn.functional as F
import torch.distributed as dist
from src.quantization import *
from src.utils.progress_reporter import report_gumbel_epoch, report_gumbel_step, report_pipeline

class QuantizationTrainer:
    def __init__(self, model, config, dtype, self_attn=False):
        self.model = model
        self.config = config
        self.quantizers = {}
        self.initialized_weights = {}
        self.optimizer = None
        self.scheduler = None
        self.device = model.device
        self.dtype = dtype
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.optimizer_params = []
        self.min_loss = float('inf')
        self.routing_cache = None
        self.batch_size = self.config.data.batch_size
        self.global_rank = getattr(self.model, 'rank', 0)
        self.world_size = getattr(self.model, 'world_size', 1)
        self.use_dist = self.world_size > 1
        if hasattr(self.model, 'save_dir'):
            self.model.save_dir = config.training.checkpoint_dir
        self.train_attn = self_attn
        
    def _create_quantizer(self, Q, scales):
        gsq_bits = getattr(self.config.quantization, 'gsq_bits', 2)
        groupsize = self.config.quantization.groupsize
        std = self.config.quantization.std
        strength = self.config.quantization.strength

        logits_dtype_str = getattr(self.config.quantization, 'logits_dtype', None)
        if logits_dtype_str == "float32":
            logits_dtype = torch.float32
        else:
            logits_dtype = self.dtype

        if is_nowag_state(Q):
            return NoWagD1GumbelQuantizer(
                Q, self.config, self.device, self.dtype, logits_dtype
            )

        if gsq_bits == 1:
            return GumbelQuantizer1Bit(Q, scales, groupsize, std, strength, self.device, self.dtype, logits_dtype)
        elif gsq_bits == 2:
            return GumbelQuantizer2Bit(Q, scales, groupsize, std, strength, self.device, self.dtype, logits_dtype)
        elif gsq_bits in [3, 4]:
            return GumbelQuantizerInt(Q, scales, groupsize, std, strength, self.device, self.dtype, logits_dtype, bits=gsq_bits)
        elif gsq_bits in ("ternary", "1.58", 1.58):
            return GumbelQuantizerTernary(Q, scales, groupsize, std, strength, self.device, self.dtype, logits_dtype)
        else:
            raise ValueError(
                f"Unsupported gsq_bits={gsq_bits!r}. "
                f"Supported: 1, 2, 3, 4, 'ternary' (aliases: '1.58')"
            )

    def setup_layer_training(self, tensor_name, Q, scales):
        if is_nowag_state(Q):
            self.initialized_weights[tensor_name] = Q
        else:
            self.initialized_weights[tensor_name] = (Q.detach(), scales.detach())
        quantizer = self._create_quantizer(Q, scales)

        if self.use_dist and not self.model.is_moe:
            for p in quantizer.parameters():
                dist.broadcast(p.data, src=0)

        self.quantizers[tensor_name] = quantizer

        if isinstance(quantizer, NoWagD1GumbelQuantizer):
            self.optimizer_params.append({
                'params': [quantizer.quant_logits],
                'lr': self.config.training.lr1,
                'weight_decay': self.config.training.weight_decay,
                'lr_decay_tag': True,
            })
            if isinstance(quantizer.log_scale, nn.Parameter):
                self.optimizer_params.append({
                    'params': [quantizer.log_scale],
                    'lr': self.config.training.lr2,
                    'weight_decay': 0.0,
                    'lr_decay_tag': True,
                })
            if quantizer.codebook_trainable:
                self.optimizer_params.append({
                    'params': [quantizer.codebook_base, quantizer.codebook_gap_raw],
                    'lr': self.config.training.nowag_codebook_lr,
                    'weight_decay': 0.0,
                    'lr_decay_tag': True,
                })
            if quantizer.norm_trainable:
                self.optimizer_params.append({
                    'params': [quantizer.row_log_ratio, quantizer.col_log_ratio],
                    'lr': self.config.training.nowag_norm_lr,
                    'weight_decay': 0.0,
                    'lr_decay_tag': True,
                })
        else:
            logit_params = [p for n, p in quantizer.named_parameters() if n != 'scales']
            self.optimizer_params.extend([
                {'params': logit_params, 'lr': self.config.training.lr1,
                'weight_decay': self.config.training.weight_decay, 'lr_decay_tag': True},
                {'params': quantizer.scales, 'lr': self.config.training.lr2,
                'weight_decay': 0.0, 'lr_decay_tag': True}
            ])

    def save_initialized_layer(self):
        """Persist the exact GPTQ/RTN initialization without GSQ training."""
        if self.train_attn:
            for tensor_name, pair in self.initialized_weights.items():
                self.model.update_quantized_weights(tensor_name, pair)
            return

        grouped = {}
        for tensor_name, pair in self.initialized_weights.items():
            base, leaf = tensor_name.rsplit(".", 1)
            grouped.setdefault(base, {})[leaf] = pair
        for base, pairs in grouped.items():
            self.model.save_to_disc(base, pairs)
        
    def train_layer(self, layer_name, train_all, val_all, logging,
                    layer_idx=None, num_layers=None):
        if logging is not None:
            logging = logging.logger
        if not self.quantizers:
            if self.global_rank == 0:
                logging.info(f"Skipping GSQ training for layer {layer_name} since no quantizers were set up.")
            return
        num_epochs = self.config.training.num_epochs
        num_samples = train_all['input'].shape[0]
        batch_size = self.config.data.batch_size // self.world_size

        self.optimizer = Lion(self.optimizer_params, betas=tuple(self.config.training.lion_betas))

        num_training_steps = (num_samples + batch_size - 1) // batch_size * num_epochs
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        self.scheduler = CustomLRScheduler(
            self.optimizer, num_training_steps,
            self.config.training.warmup_steps,
            lr_decay_type=self.config.training.lr_decay_type,
            min_lr=self.config.training.scheduler_min_lr,
        )
        
        initial_temperature, final_temperature = self.config.quantization.temperature
        initial_scale, final_scale = self.config.quantization.scale

        step = 0
        phase_start = time.time()
        step_report_interval = max(1, steps_per_epoch // self.config.logging.step_report_divisor)

        micro = max(1, self.config.training.device_microbatch_size)
        with torch.no_grad():
            initial_soft_loss, initial_hard_loss = self._average_validation_loss(
                val_all['input'], batch_size, initial_temperature, initial_scale, micro
            )
        selected_hard_loss = initial_hard_loss
        selected_states = self._snapshot_hard_states()

        for epoch in range(num_epochs):
            for quantizer in self.quantizers.values():
                if isinstance(quantizer, NoWagD1GumbelQuantizer):
                    quantizer.set_epoch(epoch)
            t = epoch / (num_epochs - 1) if num_epochs > 1 else 0.0
            temperature = initial_temperature + (final_temperature - initial_temperature) * t
            scale = initial_scale + (final_scale - initial_scale) * t

            epoch_losses = []
            epoch_start = time.time()

            for batch in self.get_random_batch_indices(train_all['input'], num_samples, batch_size):
                temperature = initial_temperature + (final_temperature - initial_temperature) * step / (num_training_steps - 1)
                scale = initial_scale + (final_scale - initial_scale) * step / (num_training_steps - 1)
                loss, micro = self._train_step_with_oom_backoff(
                    batch,
                    temperature, 
                    scale,
                    micro,
                    logging=logging,
                    layer_name=layer_name,
                )
                epoch_losses.append(loss)

                if self.global_rank == 0:
                    report_gumbel_step(step, num_training_steps, loss,
                                       interval=step_report_interval)

                if self.config.wandb.enabled and self.global_rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        "train/step_loss": loss,
                        "train/learning_rate": current_lr,
                        "train/temperature": temperature,
                        "train/scale": scale,
                        "train/global_step": step,
                    })

                step += 1

            val_soft_losses, val_hard_losses = [], []
            
            num_batches = (val_all['input'].shape[0] + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    start_idx = batch_idx*batch_size
                    end_idx = min((batch_idx+1)*batch_size,val_all['input'].shape[0])
                    
                    val_soft_loss, val_hard_loss = self.validation_step(
                        val_all['input'][start_idx:end_idx],
                        temperature, 
                        scale,
                        micro
                    )
                    val_soft_losses.append(val_soft_loss)
                    val_hard_losses.append(val_hard_loss)
            
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val_soft_loss = sum(val_soft_losses) / len(val_soft_losses)
            avg_val_hard_loss = sum(val_hard_losses) / len(val_hard_losses)
            if avg_val_hard_loss < selected_hard_loss:
                selected_hard_loss = avg_val_hard_loss
                selected_states = self._snapshot_hard_states()
            epoch_time = time.time() - epoch_start
            phase_elapsed = time.time() - phase_start

            if self.global_rank == 0:
                logging.info(
                    f'Layer {layer_name} - Epoch {epoch+1}: '
                    f'Train Loss = {avg_train_loss:.2e}, '
                    f'Val Soft Loss = {avg_val_soft_loss:.2e}, '
                    f'Val Hard Loss = {avg_val_hard_loss:.2e}'
                )
                report_gumbel_epoch(
                    layer_name, epoch, num_epochs, phase_elapsed,
                    avg_train_loss=avg_train_loss,
                    avg_val_loss=avg_val_hard_loss,
                    epoch_time=epoch_time,
                    temperature=temperature,
                    scale=scale,
                )
            
            if self.config.wandb.enabled and self.global_rank == 0:
                wandb.log({
                    f"{layer_name}/train_loss": avg_train_loss,
                    f"{layer_name}/val_soft_loss": avg_val_soft_loss,
                    f"{layer_name}/val_hard_loss": avg_val_hard_loss,
                    f"{layer_name}/temperature": temperature,
                    f"{layer_name}/scale": scale,
                    f"{layer_name}/epoch": epoch + 1,
                    f"{layer_name}/epoch_time_sec": epoch_time,
                })

            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        if self.use_dist:
            if logging is not None and self.global_rank == 0:
                report_pipeline(f"Syncing ranks before writing checkpoints ({layer_name})")
            dist.barrier()

        if logging is not None and self.global_rank == 0:
            logging.info(
                "Layer %s hard-state selection: initial_val=%.6e selected_val=%.6e accepted_training=%s",
                layer_name,
                initial_hard_loss,
                selected_hard_loss,
                selected_hard_loss < initial_hard_loss,
            )
            if self.train_attn:
                report_pipeline(f"Applying quantized attention weights in-place ({layer_name})")
            else:
                report_pipeline(f"Writing quantized shard(s) to disk ({layer_name})")

        for tensor_name, quantizer in self.quantizers.items():
            selected_state = selected_states[tensor_name]
            if self.train_attn:
                self.model.update_quantized_weights(tensor_name, selected_state)
            else:
                if "gate_proj" in tensor_name:
                    base = tensor_name[: -len(".gate_proj")]
                    pairs = {
                        "gate_proj": selected_states[tensor_name],
                        "up_proj": selected_states[f"{base}.up_proj"],
                        "down_proj": selected_states[f"{base}.down_proj"],
                    }
                    if self.model.is_moe or self.global_rank == 0:
                        self.model.save_to_disc(base, pairs)

        if logging is not None and self.global_rank == 0:
            report_pipeline(f"Finished writing layer checkpoint ({layer_name})")

        if self.use_dist:
            if logging is not None and self.global_rank == 0:
                report_pipeline("Shard writes done; syncing ranks before next phase")
            dist.barrier()
        self.quantizers.clear()
        self.initialized_weights.clear()

        del self.optimizer, self.optimizer_params, self.quantizers
        torch.cuda.empty_cache()
        
    def train_step(self, batch, temperature, scale, microbatch_size):
        self.optimizer.zero_grad(set_to_none=True)

        batch_size, seq_len, hidden_dim = batch.shape
        accumulation_steps = max(1, batch_size // microbatch_size)

        total_loss = 0.0

        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size]

            quantized_weights = {}
            for tensor_name, quantizer in self.quantizers.items():
                if self.use_dist and self.model.is_moe:
                    prefix, leaf = tensor_name.rsplit(".", 1)
                    if prefix not in quantized_weights:
                        quantized_weights[prefix] = {}
                    quantized_weights[prefix][leaf] = quantizer.forward(temperature, scale)
                else:
                    quantized_weights[tensor_name] = quantizer.forward(temperature, scale)

            loss_kwargs = {"accumulation_steps": accumulation_steps}
            if getattr(self.config.training, "defer_train_loss_item_sync", False):
                loss_kwargs["defer_item"] = True
            soft_loss = self.model.calculate_mse(
                micro_batch.to(self.device),
                quantized_weights,
                self.train_attn,
                **loss_kwargs,
            )
            total_loss += soft_loss / accumulation_steps

        self.scheduler.step()
        if not self.model.is_moe and self.use_dist:
            self.average_grads()
        self.optimizer.step()
        for quantizer in self.quantizers.values():
            if isinstance(quantizer, NoWagD1GumbelQuantizer):
                quantizer.project_parameters()

        quantized_weights.clear()

        if self.use_dist:
            pg = dist.group.WORLD
            t = torch.tensor(total_loss, device=self.device, dtype=torch.float32)
            if self.model.is_moe:
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pg)
            else:
                dist.all_reduce(t, op=dist.ReduceOp.AVG, group=pg)
            total_loss = t.item()
        
        if isinstance(total_loss, torch.Tensor):
            return total_loss.item()
        return total_loss

    @staticmethod
    def _is_cuda_oom(exc):
        # CUDA OOM has historically been raised as RuntimeError.  Newer
        # PyTorch releases expose torch.OutOfMemoryError, but the remote
        # training image intentionally uses an older compatible build.
        message = str(exc).lower()
        return isinstance(exc, RuntimeError) and (
            "cuda out of memory" in message
            or ("out of memory" in message and "cuda" in message)
        )

    def _clear_failed_step(self):
        """Drop partial gradients and cached allocations after a failed step."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _train_step_with_oom_backoff(
        self,
        batch,
        temperature,
        scale,
        microbatch_size,
        logging=None,
        layer_name=None,
    ):
        """Run one logical batch, reducing only the device microbatch on OOM.

        The optimizer and scheduler are advanced only by a successful
        ``train_step``.  An OOM therefore retries the same logical batch after
        clearing partial gradients instead of silently skipping data.
        """
        microbatch_size = max(1, min(int(microbatch_size), int(batch.shape[0])))
        allow_backoff = bool(
            getattr(self.config.training, "oom_microbatch_backoff", False)
        )
        min_microbatch = max(
            1,
            int(getattr(self.config.training, "min_device_microbatch_size", 1)),
        )

        while True:
            try:
                return (
                    self.train_step(batch, temperature, scale, microbatch_size),
                    microbatch_size,
                )
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc):
                    raise
                self._clear_failed_step()
                if not allow_backoff or microbatch_size <= min_microbatch:
                    raise

                next_microbatch = max(min_microbatch, microbatch_size // 2)
                if next_microbatch >= microbatch_size:
                    raise
                if logging is not None and self.global_rank == 0:
                    logging.warning(
                        "CUDA OOM in layer %s at device_microbatch_size=%d; "
                        "retrying the same logical batch with %d",
                        layer_name,
                        microbatch_size,
                        next_microbatch,
                    )
                microbatch_size = next_microbatch

    @staticmethod
    def _clone_hard_state(state):
        if is_nowag_state(state):
            return {
                key: value.detach().to("cpu", copy=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in state.items()
            }
        return tuple(value.detach().to("cpu", copy=True) for value in state)

    def _snapshot_hard_states(self):
        return {
            tensor_name: self._clone_hard_state(quantizer.get_hard_weights())
            for tensor_name, quantizer in self.quantizers.items()
        }

    def _average_validation_loss(self, inputs, batch_size, temperature, scale, micro):
        soft_losses, hard_losses = [], []
        num_batches = (inputs.shape[0] + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, inputs.shape[0])
            soft_loss, hard_loss = self.validation_step(
                inputs[start_idx:end_idx], temperature, scale, micro
            )
            soft_losses.append(soft_loss)
            hard_losses.append(hard_loss)
        return (
            sum(soft_losses) / max(len(soft_losses), 1),
            sum(hard_losses) / max(len(hard_losses), 1),
        )

    def _build_weights(self, mode, temperature, scale):
        weights = {}
        for tensor_name, quantizer in self.quantizers.items():
            if mode == 'soft':
                w = quantizer.forward(temperature, scale)
            else:
                hard = quantizer.get_hard_weights()
                w = hard["weight"] if is_nowag_state(hard) else hard[0]
            if self.use_dist and self.model.is_moe:
                prefix, leaf = tensor_name.rsplit(".", 1)
                if prefix not in weights:
                    weights[prefix] = {}
                weights[prefix][leaf] = w
            else:
                weights[tensor_name] = w
        return weights

    def validation_step(self, batch, temperature, scale, microbatch_size):
        batch_size, seq_len, hidden_dim = batch.shape
        microbatch_size = max(1, microbatch_size)
        accumulation_steps = max(1, batch_size // microbatch_size)

        total_soft_loss = 0.0
        total_hard_loss = 0.0

        soft_weights = self._build_weights('soft', temperature, scale)
        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size].to(self.device)
            total_soft_loss += self.model.calculate_mse(micro_batch, soft_weights, self.train_attn, validation=True) / accumulation_steps
        soft_weights.clear()

        hard_weights = self._build_weights('hard', temperature, scale)
        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size].to(self.device)
            total_hard_loss += self.model.calculate_mse(micro_batch, hard_weights, self.train_attn, validation=True) / accumulation_steps
        hard_weights.clear()

        if self.use_dist:
            pg = dist.group.WORLD
            t_soft = torch.tensor(total_soft_loss, device=self.device, dtype=torch.float32)
            t_hard = torch.tensor(total_hard_loss, device=self.device, dtype=torch.float32)
            if self.model.is_moe:
                dist.all_reduce(t_soft, op=dist.ReduceOp.SUM, group=pg)
                dist.all_reduce(t_hard, op=dist.ReduceOp.SUM, group=pg)
            else:
                dist.all_reduce(t_soft, op=dist.ReduceOp.AVG, group=pg)
                dist.all_reduce(t_hard, op=dist.ReduceOp.AVG, group=pg)
            total_soft_loss = t_soft.item()
            total_hard_loss = t_hard.item()

        return total_soft_loss, total_hard_loss
    
    def get_random_batch_indices(self, x, num_samples, batch_size):
        perm = torch.randperm(num_samples // batch_size)
        for i in range(num_samples // batch_size):
            start = perm[i] * batch_size
            yield x[start:start+batch_size]

    def average_grads(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(self.world_size)

class CustomLRScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps, lr_decay_type='linear', min_lr=0.0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_decay_type = lr_decay_type
        self.min_lr = min_lr
        self.current_step = 0

        self.initial_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        for i, group in enumerate(self.optimizer.param_groups):
            tag = group.get('lr_decay_tag')
            init_lr = self.initial_lrs[i]

            if tag:
                group['lr'] = self._compute_lr(init_lr)
        
        self.current_step += 1

    def _compute_lr(self, base_lr):
        step = self.current_step
        if step < self.warmup_steps:
            return base_lr * (self.min_lr + (1 - self.min_lr) * step / self.warmup_steps)

        decay_step = step - self.warmup_steps
        decay_total = self.total_steps - 1 - self.warmup_steps
        if decay_total == 0:
            progress = 1
        else:
            progress = decay_step / decay_total

        if self.lr_decay_type == 'linear':
            return base_lr * (self.min_lr + (1 - self.min_lr) * (1 - progress))
        elif self.lr_decay_type == 'cosine':
            return base_lr * (self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * progress)))
        elif self.lr_decay_type == 'constant':
            return base_lr
        else:
            raise ValueError(f"Unknown lr_decay_type: {self.lr_decay_type}")
