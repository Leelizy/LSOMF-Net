"""OCGC correction, seasonal lookup tables and channel attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_ST


class ObservationContext(nn.Module):
    """Encode month, missingness ratio, and correction-stage context."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.month_emb = nn.Embedding(12, 2)
        self.mask_proj = nn.Linear(1, 1)
        self.future_emb = nn.Embedding(2, 1)
        self.fuse = nn.Linear(4, dim)

    def forward(
        self,
        month: torch.Tensor,
        mask_ratio: float | torch.Tensor,
        is_future: bool = False,
    ) -> torch.Tensor:
        batch = month.shape[0]
        if month.dim() == 2:
            month = month[:, 0]
        month = month.long()
        month_code = self.month_emb(month)
        mask_code = torch.as_tensor(
            mask_ratio, device=month.device, dtype=torch.float32
        )
        if mask_code.numel() == 1:
            mask_code = mask_code.reshape(1, 1).expand(batch, 1)
        else:
            mask_code = mask_code.reshape(batch, 1)
        mask_code = self.mask_proj(mask_code if not is_future else torch.zeros_like(mask_code))
        future_flag = torch.full(
            (batch,), int(is_future), device=month.device, dtype=torch.long
        )
        future_code = self.future_emb(future_flag)
        return self.fuse(torch.cat([month_code, mask_code, future_code], dim=-1))


class OCGC(nn.Module):
    """Blend corrected and observed inputs using the observation mask."""

    def __init__(
        self,
        n_st: int = N_ST,
        ctx_dim: int = 4,
    ):
        super().__init__()
        self.ctx_encoder = ObservationContext(ctx_dim)
        input_channels = 2 * n_st + 1
        self.feature_in = nn.Conv3d(input_channels, n_st, 1)
        self.feature_gate = nn.Conv3d(input_channels, n_st, 1)
        self.temporal_dw = nn.Conv3d(
            n_st, n_st, (3, 1, 1), padding=(1, 0, 0), groups=n_st
        )
        self.spatial_dw = nn.Conv3d(
            n_st, n_st, (1, 3, 3), padding=(0, 1, 1), groups=n_st
        )
        self.channel_mix = nn.Conv3d(n_st, n_st, 1)
        self.delta_out = nn.Conv3d(n_st, n_st, 1)
        self.prior_gate = nn.Conv3d(n_st + 1, 1, 1)
        self.prior_mix = nn.Conv3d(n_st + 1, 1, 1)
        self.prior_mix_ctx = nn.Linear(ctx_dim, 1)
        self.ctx_gate = nn.Linear(ctx_dim, n_st)
        # Initialize equal input/baseline weights and zero residual correction.
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)
        nn.init.zeros_(self.prior_gate.weight)
        nn.init.constant_(self.prior_gate.bias, -1.0)
        nn.init.zeros_(self.prior_mix.weight)
        nn.init.zeros_(self.prior_mix.bias)
        nn.init.zeros_(self.prior_mix_ctx.weight)
        nn.init.zeros_(self.prior_mix_ctx.bias)

    @staticmethod
    def _valid_mask(
        x: torch.Tensor,
        mask_ratio: float | torch.Tensor,
        spatial_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, steps, _, height, width = x.shape
        if spatial_mask is not None:
            if spatial_mask.shape != (batch, steps, height, width):
                raise ValueError(
                    "spatial_mask must have shape [B,T,H,W], got "
                    f"{tuple(spatial_mask.shape)}"
                )
            return spatial_mask.to(dtype=x.dtype).unsqueeze(2).clamp(0.0, 1.0)

        ratio = torch.as_tensor(mask_ratio, device=x.device, dtype=x.dtype)
        if ratio.numel() == 1:
            return torch.full(
                (batch, steps, 1, height, width),
                1.0 - float(ratio.item()),
                device=x.device,
                dtype=x.dtype,
            )
        ratio = ratio.reshape(batch, 1, 1, 1, 1)
        return (1.0 - ratio).expand(batch, steps, 1, height, width).clamp(0.0, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        month: torch.Tensor,
        mask_ratio: float | torch.Tensor = 0.0,
        spatial_mask: torch.Tensor | None = None,
        baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        if baseline is None:
            baseline = x
        if baseline.shape != x.shape:
            raise ValueError(
                f"baseline must match x shape {tuple(x.shape)}, got {tuple(baseline.shape)}"
            )
        valid = self._valid_mask(x, mask_ratio, spatial_mask)
        context_input = torch.cat([x, baseline, valid], dim=2)
        context_input = context_input.permute(0, 2, 1, 3, 4).contiguous()
        hidden = F.gelu(self.feature_in(context_input))
        hidden = hidden * torch.sigmoid(self.feature_gate(context_input))
        hidden = F.gelu(self.temporal_dw(hidden))
        hidden = F.gelu(self.spatial_dw(hidden))
        hidden = self.channel_mix(hidden)
        if month is not None:
            context = self.ctx_encoder(month, mask_ratio, is_future=False)
            gate = self.ctx_gate(context)
            hidden = hidden * (1.0 + torch.tanh(gate).view(batch, channels, 1, 1, 1))
        delta = self.delta_out(hidden)
        prior_gate_input = torch.cat([hidden, valid.permute(0, 2, 1, 3, 4)], dim=1)
        prior_gate = torch.sigmoid(self.prior_gate(prior_gate_input))
        prior_mix_logits = self.prior_mix(prior_gate_input)
        if month is not None:
            prior_mix_logits = prior_mix_logits + self.prior_mix_ctx(context).view(
                batch, 1, 1, 1, 1
            )
        prior_mix = torch.sigmoid(prior_mix_logits)
        neighbor_prior = x.permute(0, 2, 1, 3, 4)
        seasonal_prior = baseline.permute(0, 2, 1, 3, 4)
        prior = prior_mix * neighbor_prior + (1.0 - prior_mix) * seasonal_prior
        candidate = prior + prior_gate * delta
        candidate = candidate.permute(0, 2, 1, 3, 4)
        # Blend the input and corrected candidate using the observation mask.
        return valid * x + (1.0 - valid) * candidate


class CycleTable(nn.Module):
    """Learnable table for a period-specific latent seasonal component."""

    def __init__(self, period: int, n_vars: int):
        super().__init__()
        self.data = nn.Parameter(torch.zeros(period, n_vars))

    def forward(self, indices: torch.Tensor, steps: int) -> torch.Tensor:
        offsets = torch.arange(steps, device=indices.device)
        idx = (indices.unsqueeze(1) + offsets) % self.data.shape[0]
        return self.data[idx].permute(0, 2, 1)


class ChannelAttention(nn.Module):
    """Lightweight squeeze-excitation across latent channels."""

    def __init__(self, n_vars: int = N_ST, reduction: int = 4):
        super().__init__()
        reduced = max(2, (n_vars + reduction - 1) // reduction)
        self.fc1 = nn.Linear(n_vars, reduced)
        self.fc2 = nn.Linear(reduced, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(-1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale.unsqueeze(-1)


def param_count(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)
