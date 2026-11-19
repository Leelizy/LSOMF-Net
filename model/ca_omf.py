"""Cycle-Aware Orthogonal Moment Fusion (CA-OMF) predictor."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import H_OUT, N_ST, T_IN
from model.model import ChannelAttention, CycleTable


class _CAOMFBase(nn.Module):
    """Seasonal preprocessing and spatial flatten/restore utilities."""

    def __init__(self, n_vars: int, t_in: int = T_IN):
        super().__init__()
        self.cycle12 = CycleTable(12, n_vars)
        self.cycle6 = CycleTable(6, n_vars)
        self.se = ChannelAttention(n_vars)
        self.n_vars = n_vars
        self.t_in = t_in

    def _preprocess(self, x: torch.Tensor, month: torch.Tensor):
        batch, t_in, channels, height, width = x.shape
        flat = (
            x.permute(0, 3, 4, 1, 2)
            .reshape(batch * height * width, t_in, channels)
            .permute(0, 2, 1)
        )
        # Convert the last input month to the first input month.
        last_month_grid = month.unsqueeze(1).expand(batch, height * width).reshape(-1)
        start_month_grid = (last_month_grid - (t_in - 1)) % 12
        cycle12 = self.cycle12(start_month_grid, t_in)
        cycle6 = self.cycle6(start_month_grid % 6, t_in)
        residual = flat - cycle12 - cycle6
        return self.se(residual), batch, height, width

    def _postprocess(
        self,
        pred_residual: torch.Tensor,
        month: torch.Tensor,
        batch: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        # The forecast starts one month after the last input.
        last_month_grid = month.unsqueeze(1).expand(batch, height * width).reshape(-1)
        future_month_grid = (last_month_grid + 1) % 12
        cycle12 = self.cycle12(future_month_grid, H_OUT)
        cycle6 = self.cycle6(future_month_grid % 6, H_OUT)
        prediction = (pred_residual + cycle12 + cycle6).reshape(
            batch, height, width, -1, H_OUT
        )
        return prediction.permute(0, 4, 3, 1, 2).contiguous()


class PhaseObservationAdapter(nn.Module):
    """Month/missingness modulation with zero-initialized gain."""

    def __init__(self, n_vars: int):
        super().__init__()
        self.phase_embedding = nn.Embedding(12, n_vars)
        self.mask_proj = nn.Linear(1, n_vars)
        self.gain = nn.Parameter(torch.zeros(n_vars))

    def forward(
        self,
        x: torch.Tensor,
        month: torch.Tensor,
        mask_ratio: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        batch = month.shape[0]
        mask = torch.as_tensor(mask_ratio, device=x.device, dtype=x.dtype)
        if mask.numel() == 1:
            mask = mask.reshape(1, 1).expand(batch, 1)
        else:
            mask = mask.reshape(batch, 1)
        condition = self.phase_embedding(month.long()) + self.mask_proj(mask)
        scale = self.gain.unsqueeze(0) * torch.tanh(condition)
        return x * (1.0 + scale.unsqueeze(-1))


@dataclass
class CAOMFConfig:
    """Configuration for the CA-OMF predictor."""

    name: str = "lsomf_ca_omf"
    e_dim: int = 6
    f_hidden: int = 24
    n_moments: int = 4


class CAOMF(nn.Module):
    """Cycle-Aware Orthogonal Moment Fusion (CA-OMF) predictor."""

    def __init__(
        self,
        n_vars: int = N_ST,
        t_in: int = T_IN,
        cfg: CAOMFConfig | None = None,
    ):
        super().__init__()
        self.base = _CAOMFBase(n_vars, t_in)
        self.cfg = cfg or CAOMFConfig()
        cfg = self.cfg

        self.phase_adapter = PhaseObservationAdapter(n_vars)
        self.G = nn.Linear(n_vars, cfg.e_dim)

        # QR orthogonalization of polynomial lag moments.
        lag_count = t_in - 1
        r = torch.linspace(-1.0, 1.0, lag_count).view(1, lag_count)
        powers = torch.arange(cfg.n_moments).view(-1, 1)
        vandermonde = r**powers
        q, _ = torch.linalg.qr(vandermonde.T)
        self.register_buffer("B", q.T.contiguous())

        input_dim = (cfg.n_moments + 1) * cfg.e_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, cfg.f_hidden),
            nn.GELU(),
            nn.Linear(cfg.f_hidden, n_vars),
        )
        nn.init.normal_(self.fusion[-1].weight, std=0.01)
        nn.init.zeros_(self.fusion[-1].bias)

        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        month: torch.Tensor,
        mask_ratio: float | torch.Tensor = 0.0,
    ) -> torch.Tensor:
        residual, batch, height, width = self.base._preprocess(x, month)
        month_grid = month.repeat_interleave(height * width)
        residual = self.phase_adapter(residual, month_grid, mask_ratio)

        flat_batch = residual.shape[0]
        terminal = residual[:, :, -1]
        history = torch.flip(residual[:, :, :-1], dims=[-1])[:, :, : self.B.shape[1]]

        projected_history = self.G(history.permute(0, 2, 1))
        terminal_projection = self.G(terminal)
        moments = torch.einsum("jk,nkl->njl", self.B, projected_history)
        features = [
            moments.reshape(flat_batch, self.cfg.n_moments * self.cfg.e_dim),
            terminal_projection,
        ]

        update = self.fusion(torch.cat(features, dim=-1))
        prediction_residual = terminal.unsqueeze(-1) + self.out_scale * update.unsqueeze(-1)
        return self.base._postprocess(
            prediction_residual, month, batch, height, width
        )


FINAL_CFG = CAOMFConfig()
