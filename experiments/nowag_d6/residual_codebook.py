"""Hard codebook weights with explicitly budgeted FP16 low-rank correction."""

import torch
from torch import nn

from experiments.nowag_d6.hard_codebook import HardCodebookWeight, _round_fp16_st


class ResidualCodebookWeight(HardCodebookWeight):
    def __init__(self, logical_state, *, rank):
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError('Residual rank must be a positive integer')
        keys = {'residual_A', 'residual_B'}
        present = keys.intersection(logical_state)
        if present and present != keys:
            raise ValueError('Residual checkpoint must contain both factors')
        super().__init__({key: value for key, value in logical_state.items() if key not in keys})
        if rank > min(self._rows, self._columns):
            raise ValueError('Residual rank exceeds weight dimensions')
        self.rank = rank
        shapes = {'residual_A': (rank, self._columns), 'residual_B': (self._rows, rank)}
        for name, shape in shapes.items():
            if present:
                value = logical_state[name]
                if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float16:
                    raise ValueError('Residual factor shape or storage dtype mismatch')
                self._validate_finite(name, value)
                value = value.detach().float().clone()
            else:
                value = torch.randn(shape, dtype=torch.float32) * .01 if name == 'residual_A' else torch.zeros(shape)
            setattr(self, name, nn.Parameter(value))

    @property
    def residual_bits(self):
        return 16 * self.rank * (self._rows + self._columns)

    def _factors(self):
        a, b = _round_fp16_st(self.residual_A), _round_fp16_st(self.residual_B)
        self._validate_finite('residual_A', a)
        self._validate_finite('residual_B', b)
        return a, b

    def forward(self):
        a, b = self._factors()
        return super().forward() + b @ a

    def export_state(self):
        state = super().export_state()
        a, b = self._factors()
        state.update(residual_A=a.detach().to('cpu', torch.float16),
                     residual_B=b.detach().to('cpu', torch.float16))
        return state
