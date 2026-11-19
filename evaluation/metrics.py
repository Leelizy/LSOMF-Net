"""Forecast accuracy and spatial error summaries for the main experiment."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config import H_OUT, T_IN
from data_preprocessing.prepare import decode_to_raw, split_indices

QUADRANTS = {
    "SW": (0, 8, 0, 8),
    "SE": (0, 8, 8, 16),
    "NW": (8, 16, 0, 8),
    "NE": (8, 16, 8, 16),
}


def evaluate_final(
    model: nn.Module,
    data: dict[str, object],
    eval_split: str = "val",
) -> dict[str, object]:
    """Average per-window forecast metrics; also report pooled alternatives."""

    indices = split_indices(data, eval_split)
    model.eval()
    device = next(model.parameters()).device
    metric_lists: dict[str, list[list[float]]] = {
        "rmse": [[] for _ in range(H_OUT)],
        "mae": [[] for _ in range(H_OUT)],
        "mape": [[] for _ in range(H_OUT)],
        "r2": [[] for _ in range(H_OUT)],
    }
    quadrants: dict[str, list[float]] = {name: [] for name in QUADRANTS}
    total_sse = 0.0
    total_ae = 0.0
    total_ape = 0.0
    total_truth_sum = 0.0
    total_truth_sq = 0.0
    total_count = 0

    with torch.no_grad():
        for index_value in indices:
            index = int(index_value)
            _, forecast = model(
                data["xs"][index : index + 1].to(device),
                data["ms"][index : index + 1].to(device),
                mask_ratio=data["mask_ratio"],
                spatial_mask=data["mw"][index : index + 1].to(device),
                baseline=data["xb"][index : index + 1].to(device),
            )
            for lead in range(H_OUT):
                target_time = index + T_IN + lead
                if target_time >= len(data["svel_raw"]):
                    continue
                prediction = decode_to_raw(forecast[:, lead], data)
                truth = data["svel_raw"][target_time].unsqueeze(0).to(device)
                difference = prediction - truth
                target = truth[0]

                sse = torch.sum(difference.square())
                ae = torch.sum(difference.abs())
                ape = torch.sum(
                    difference.abs() / torch.clamp(target.abs(), min=1e-6)
                )
                sst = torch.sum((target - target.mean()).square())
                r2 = (
                    1.0 - sse / sst
                ).item() if sst.item() > 1e-12 else float("nan")
                metric_lists["rmse"][lead].append(
                    float(torch.sqrt(sse / difference.numel()).item())
                )
                metric_lists["mae"][lead].append(
                    float((ae / difference.numel()).item())
                )
                metric_lists["mape"][lead].append(
                    float((100.0 * ape / difference.numel()).item())
                )
                metric_lists["r2"][lead].append(float(r2))

                total_sse += float(sse.item())
                total_ae += float(ae.item())
                total_ape += float(ape.item())
                total_truth_sum += float(target.sum().item())
                total_truth_sq += float(target.square().sum().item())
                total_count += int(difference.numel())

                if lead == 0:
                    for name, (hs, he, ws, we) in QUADRANTS.items():
                        quadrants[name].append(
                            float(
                                torch.sqrt(
                                    torch.mean(
                                        difference[0, :, hs:he, ws:we].square()
                                    )
                                ).item()
                            )
                        )

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    per_lead = [mean(values) for values in metric_lists["rmse"]]
    mae_per_lead = [mean(values) for values in metric_lists["mae"]]
    mape_per_lead = [mean(values) for values in metric_lists["mape"]]
    r2_per_lead = [mean(values) for values in metric_lists["r2"]]
    pooled_sst = (
        total_truth_sq - (total_truth_sum * total_truth_sum) / total_count
        if total_count
        else 0.0
    )
    return {
        "evaluation_split": eval_split,
        "lead1": per_lead[0] if per_lead else 0.0,
        "all_lead": mean([value for values in metric_lists["rmse"] for value in values]),
        "per_lead": per_lead,
        "mae_all_lead": mean([value for values in metric_lists["mae"] for value in values]),
        "mae_per_lead": mae_per_lead,
        "mape_all_lead": mean([value for values in metric_lists["mape"] for value in values]),
        "mape_per_lead": mape_per_lead,
        "r2_all_lead": mean([value for values in metric_lists["r2"] for value in values]),
        "r2_per_lead": r2_per_lead,
        "pooled_rmse": float(np.sqrt(total_sse / total_count)) if total_count else 0.0,
        "pooled_mae": float(total_ae / total_count) if total_count else 0.0,
        "pooled_mape": float(100.0 * total_ape / total_count) if total_count else 0.0,
        "pooled_r2": float(1.0 - total_sse / pooled_sst) if pooled_sst > 1e-12 else float("nan"),
        "quad": {name: mean(values) for name, values in quadrants.items()},
        "quad_gap": (
            max(mean(values) for values in quadrants.values())
            - min(mean(values) for values in quadrants.values())
            if quadrants
            else 0.0
        ),
    }
