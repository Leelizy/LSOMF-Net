"""Model factory for the released LSOMF-Net implementation."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import N_ST, T_IN
from model.model import OCGC, param_count
from model.ca_omf import CAOMF, CAOMFConfig, FINAL_CFG

MODEL_VERSION = "lsomf-net-fixed-v6-dual-prior-ocgc-calendar-aligned"


class LSOMFNet(nn.Module):
    """Connect OCGC correction and CA-OMF prediction after ACPE preprocessing."""

    def __init__(
        self,
        n_st: int = N_ST,
        t_in: int = T_IN,
        cfg: CAOMFConfig | None = None,
    ):
        super().__init__()
        self.ocgc = OCGC(n_st=n_st, ctx_dim=4)
        self.ca_omf = CAOMF(n_vars=n_st, t_in=t_in, cfg=cfg or FINAL_CFG)

    def forward(
        self,
        x: torch.Tensor,
        month: torch.Tensor,
        mask_ratio: float | torch.Tensor = 0.0,
        spatial_mask: torch.Tensor | None = None,
        baseline: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        refined = self.ocgc(
            x,
            month,
            mask_ratio,
            spatial_mask=spatial_mask,
            baseline=baseline,
        )
        forecast = self.ca_omf(refined, month, mask_ratio)
        return refined, forecast


def get_model(
    n_st: int = N_ST,
    t_in: int = T_IN,
    cfg: CAOMFConfig | None = None,
) -> LSOMFNet:
    """Construct the final LSOMF-Net model."""

    return LSOMFNet(n_st=n_st, t_in=t_in, cfg=cfg)


def param_table(n_st: int = N_ST) -> str:
    """Return a compact parameter-count table for the final model."""

    lines = ["=" * 72, f"  LSOMF-Net parameter table (n_st={n_st})", "=" * 72]
    model = get_model(n_st=n_st)
    ocgc_params = param_count(model.ocgc)
    ca_omf_params = param_count(model.ca_omf)
    total = ocgc_params + ca_omf_params
    lines.append(
        f"  OCGC={ocgc_params:>6}  CA-OMF={ca_omf_params:>6}  total={total:>6}"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    print(param_table())
