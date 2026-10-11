import logging
import math
import time
import torch
import torch.nn as nn
import transformers
from .quant import *
from ..quantization import *
import torch.distributed as dist
from lion_pytorch import Lion


DEBUG = False 


def rtn_quantize(layer, config, device, dtype):
    W = layer.weight.data.clone().float()
    rows, columns = W.shape
    groupsize = config.gptq.groupsize
    wbits = config.gptq.wbits
    sym = config.gptq.sym
    trits = getattr(config.gptq, 'trits', False)

    n_groups = (columns + groupsize - 1) // groupsize if groupsize > 0 else 1
    group_scales = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)
    Q = torch.zeros_like(W)

    quantizer = Quantizer()
    quantizer.configure(wbits, perchannel=True, sym=sym, mse=True, trits=trits)

    for g in range(n_groups):
        c_start = g * groupsize
        c_end = min(c_start + groupsize, columns)
        W_group = W[:, c_start:c_end]

        quantizer.find_params(W_group, weight=True)
        group_scales[:, g] = quantizer.scale.view(rows)
        Q[:, c_start:c_end] = quantize(
            W_group, quantizer.scale, quantizer.zero, quantizer.maxq
        )

    return Q.to(device), group_scales.to(device)


class GPTQ:

    def __init__(self, layer, name, config, device, dtype):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.quantizer = None
        self.H = None
        self.dead = None
        self.nsamples = 0
        self.config = config
        self.device = device
        self.dtype = dtype

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def cholesky(self, percdamp=.01):
        if self.H is None:
            logging.getLogger(__name__).warning(
                "GPTQ %s: no calibration batches (H is None); using identity Hessian fallback", self.name
            )
            self.H = torch.eye(self.columns, device=self.dev, dtype=torch.float32)
        self.dead = torch.diag(self.H) == 0
        self.H[self.dead, self.dead] = 1

        damp = percdamp * torch.mean(torch.diag(self.H))
        diag = torch.arange(self.columns, device=self.dev)
        self.H[diag, diag] += damp
        self.H = torch.linalg.cholesky(self.H)
        self.H = self.H.to(self.dtype)

    def sync_H(self, world_size):
        dist.all_reduce(self.H, op=dist.ReduceOp.SUM)
        self.H = self.H / world_size

    def fasterquant(
        self, logging, blocksize=128, percdamp=.01, groupsize=-1, static_groups=False, calculate_cholesky=True, prunen=0, prunem=0
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        if str(self.config.quantization.init_method).lower().startswith("nowag_d1"):
            return self._fasterquant_nowag(
                W,
                logging,
                blocksize=blocksize,
                percdamp=percdamp,
                groupsize=groupsize,
            )
        if calculate_cholesky:
            self.cholesky(percdamp)
        W[:, self.dead] = 0

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        rows = W.shape[0]
        if groupsize == -1:
            n_groups = 1
        else:
            n_groups = (self.columns + groupsize - 1) // groupsize
        group_scales = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)
        group_zeros  = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)
            for g, q in enumerate(groups):
                group_scales[:, g] = q.scale.view(rows)
                group_zeros[:, g]  = q.zero.view(rows)

        gsq_bits = self.config.quantization.gsq_bits
        if gsq_bits == "24int":
            possible_masks = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=self.dtype, device=self.device)
            possible_masks = torch.repeat_interleave(self.possible_masks, 2).reshape(-1, 8)
        else:
            possible_masks = None

        if gsq_bits not in ("ternary", "1.58", 1.58):
            total_loss = torch.zeros((), device=self.dev)
            Q = torch.zeros_like(W)

            H = torch.cholesky_inverse(self.H.float())
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if prunen > 0:
                    mask1 = torch.zeros_like(W1, dtype=torch.bool)
                else:
                    mask1 = None

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if self.quantizer and groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i):(i1 + i + groupsize)],
                                    weight=True
                                )
                                g = (i1 + i) // groupsize
                                group_scales[:, g] = self.quantizer.scale.view(rows)
                                group_zeros[:, g] = self.quantizer.zero.view(rows)
                        else:
                            idx = i1 + i
                            self.quantizer = groups[idx // groupsize]

                    if possible_masks is not None and (i % prunem == 0):
                        m = self.possible_masks.shape[1]
                        if i + m > count:
                            continue

                        block = W1[:, i:(i + m)]
                        d_block = torch.diag(Hinv1)[i:(i + m)]

                        scores = (block ** 2) / (d_block.reshape(1, -1) ** 2)
                        masks = self.possible_masks
                        loss_per_mask = torch.matmul(
                            scores, (1 - masks).T
                        )
                        best_mask_idx = torch.argmin(loss_per_mask, dim=1)
                        selected_masks = masks[best_mask_idx]
                        mask1[:, i:(i + m)] = (selected_masks == 0)
                    elif prunen > 0 and (i % prunem == 0):
                        m = min(prunem, count - i)
                        n = min(prunen, m)
                        block_scores = (W1[:, i:(i + m)] ** 2) / (
                            torch.diag(Hinv1)[i:(i + m)].reshape((1, -1)) ** 2
                        )
                        idx = torch.topk(block_scores, n, dim=1, largest=False).indices
                        mask1.scatter_(1, i + idx, True)

                    q = w.clone()
                    if mask1 is not None:
                        q[mask1[:, i]] = 0

                    if self.quantizer:
                        q = quantize(
                            q.unsqueeze(1),
                            self.quantizer.scale,
                            self.quantizer.zero,
                            self.quantizer.maxq,
                        ).flatten()

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / (d ** 2)

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                total_loss += Losses1.sum() / 2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            
            Q = W.clone()
            W = self.layer.weight.data.clone()
            gptq_loss = torch.linalg.norm((W - Q.to(self.dtype)) @ self.H, ord='fro')**2
            self.last_gptq_loss = gptq_loss.item()
            if logging is not None:
                logging.info(f'Layer {self.name}: GPTQ Loss = {self.last_gptq_loss}')

        run_inline_gsq = bool(self.config.quantization.gsq_enabled) and (
            "q_proj" in self.name
            or "k_proj" in self.name
            or "in_proj_qkv" in self.name
            or gsq_bits in ("ternary", "1.58", 1.58)
        )
        if run_inline_gsq:
            if gsq_bits == 1:
                quantizer = GumbelQuantizer1Bit(Q, group_scales, groupsize, self.config.quantization.std, self.config.quantization.strength, self.device, self.dtype)
            elif gsq_bits == 2:
                quantizer = GumbelQuantizer2Bit(Q, group_scales, groupsize, self.config.quantization.std, self.config.quantization.strength, self.device, self.dtype, logits_dtype=torch.float32)
            elif gsq_bits in ("ternary", "1.58", 1.58):
                quantizer = GumbelQuantizerTernary(torch.zeros_like(W), group_scales, groupsize, self.config.quantization.std, 0, self.device, self.dtype, logits_dtype=torch.float32)
            elif gsq_bits in [3, 4]:
                quantizer = GumbelQuantizerInt(Q, group_scales, groupsize, self.config.quantization.std, self.config.quantization.strength, self.device, self.dtype, logits_dtype=torch.float32)
            else:
                raise ValueError(
                    f"Unsupported gsq_bits={gsq_bits!r}. "
                    f"Supported: 1, 2, 3, 4, 'ternary' (aliases: '1.58')"
                )

            optimizer_params = []

            logit_params = [p for n, p in quantizer.named_parameters() if n != 'scales']
            optimizer_params.extend([
                {'params': logit_params, 'lr': self.config.training.lr1,
                'weight_decay': self.config.training.weight_decay},
                {'params': quantizer.scales, 'lr': self.config.training.lr2,
                'weight_decay': 0.0}
            ])

            optimizer = Lion(optimizer_params, betas=self.config.training.lion_betas)
            num_epochs = 2000

            initial_temperature, final_temperature = self.config.quantization.temperature
            initial_scale, final_scale = self.config.quantization.scale
            if gsq_bits in ("ternary", "1.58", 1.58):
                initial_temperature *= 2
            
            for epoch in range(num_epochs):
                temperature = initial_temperature + (final_temperature - initial_temperature) * (epoch / (num_epochs - 1))
                scale = initial_scale + (final_scale - initial_scale) * (epoch / (num_epochs - 1))
                W_q = quantizer.forward(temperature, scale)
                loss = torch.linalg.norm((W - W_q) @ self.H, ord='fro')**2
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            W_q, scales = quantizer.get_hard_weights()
            final_loss = torch.linalg.norm((W - W_q) @ self.H, ord='fro')**2
            if logging is not None:
                logging.info(f'Layer {self.name}: '
                                f'Train Loss = {final_loss.item():.3f}')

            optimizer_params.clear()
            
            return W_q, scales

        return Q, group_scales

    def _fasterquant_nowag(self, W, logging, blocksize, percdamp, groupsize):
        """NoWag-normalized d=1 initialization and optional independent q/k GSQ."""
        if self.H is None:
            raw_hessian = torch.eye(self.columns, device=self.dev, dtype=torch.float32)
        else:
            raw_hessian = self.H.float().clone()

        qcfg = self.config.quantization
        state = initialize_nowag_d1(
            W,
            raw_hessian,
            groupsize=groupsize,
            hessian_weight_mode=qcfg.nowag_hessian_weight_mode,
            scale_init=qcfg.nowag_scale_init,
            codebook_aware_gptq_enabled=bool(qcfg.nowag_codebook_aware_gptq),
            seed=int(qcfg.nowag_seed),
            max_iters=int(qcfg.nowag_lloyd_iters),
            blocksize=blocksize,
            percdamp=percdamp,
            eps=float(qcfg.nowag_eps),
        )
        diagnostics = state.get("diagnostics", {})
        initial_state = state
        initial_objective = float(diagnostics.get("full_hessian_objective", 0.0))
        self.last_gptq_loss = initial_objective
        if logging is not None:
            logging.info(
                "Layer %s: NoWag-d1 init full-Hessian objective = %.6e, rel-L2 = %.6e, codebook = %s",
                self.name,
                self.last_gptq_loss,
                float(diagnostics.get("weight_relative_l2", float("nan"))),
                [round(float(x), 7) for x in state["codebook"]],
            )

        run_inline_gsq = bool(qcfg.gsq_enabled) and (
            "q_proj" in self.name or "k_proj" in self.name or "in_proj_qkv" in self.name
        )
        if not run_inline_gsq:
            return state, None

        logits_dtype = torch.float32 if qcfg.logits_dtype == "float32" else self.dtype
        quantizer = NoWagD1GumbelQuantizer(
            state, self.config, self.device, self.dtype, logits_dtype
        )
        parameter_groups = [
            {
                "params": [quantizer.quant_logits],
                "lr": self.config.training.lr1,
                "weight_decay": self.config.training.weight_decay,
            }
        ]
        if isinstance(quantizer.log_scale, nn.Parameter):
            parameter_groups.append(
                {
                    "params": [quantizer.log_scale],
                    "lr": self.config.training.lr2,
                    "weight_decay": 0.0,
                }
            )
        if quantizer.codebook_trainable:
            parameter_groups.append(
                {
                    "params": [quantizer.codebook_base, quantizer.codebook_gap_raw],
                    "lr": self.config.training.nowag_codebook_lr,
                    "weight_decay": 0.0,
                }
            )
        if quantizer.norm_trainable:
            parameter_groups.append(
                {
                    "params": [quantizer.row_log_ratio, quantizer.col_log_ratio],
                    "lr": self.config.training.nowag_norm_lr,
                    "weight_decay": 0.0,
                }
            )
        optimizer = Lion(parameter_groups, betas=tuple(self.config.training.lion_betas))

        hessian = raw_hessian.clone()
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1.0
        diag = torch.arange(self.columns, device=hessian.device)
        hessian[diag, diag] += percdamp * torch.diag(hessian).mean()
        hessian_chol = torch.linalg.cholesky(hessian)

        steps = max(1, int(qcfg.nowag_inline_steps))
        initial_temperature, final_temperature = qcfg.temperature
        initial_kappa, final_kappa = qcfg.scale
        virtual_epochs = max(1, int(self.config.training.num_epochs))
        warmup_fraction = min(
            1.0, int(qcfg.nowag_scale_warmup_epochs) / virtual_epochs
        )
        warmup_steps = int(round(steps * warmup_fraction))

        for step in range(steps):
            t = step / max(steps - 1, 1)
            temperature = initial_temperature + (final_temperature - initial_temperature) * t
            kappa = initial_kappa + (final_kappa - initial_kappa) * t
            quantizer.set_epoch(0 if step < warmup_steps else qcfg.nowag_scale_warmup_epochs)
            quantized = quantizer.forward(temperature, kappa).float()
            loss = ((W - quantized) @ hessian_chol).square().sum()
            loss.backward()
            optimizer.step()
            quantizer.project_parameters()
            optimizer.zero_grad(set_to_none=True)

        trained_state = quantizer.get_hard_weights()
        final_objective = hessian_objective(W, trained_state["weight"], raw_hessian)
        accepted = final_objective <= initial_objective * (1.0 + 1e-7)
        state = trained_state if accepted else initial_state
        self.last_gptq_loss = min(final_objective, initial_objective)
        state["diagnostics"] = {
            **quantizer.audit(),
            "full_hessian_objective": self.last_gptq_loss,
            "trained_full_hessian_objective": final_objective,
            "hard_state_accepted": float(accepted),
        }
        if logging is not None:
            logging.info(
                "Layer %s: NoWag-d1 independent GSQ objective = %.6e, accepted=%s, audit = %s",
                self.name,
                self.last_gptq_loss,
                accepted,
                state["diagnostics"],
            )
        del optimizer, quantizer, hessian_chol
        return state, None

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
