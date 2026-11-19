"""Initialize GDCSM values at Argo-observed locations."""
from __future__ import annotations

import numpy as np


def initialize_argo(values: np.ndarray, observation_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill horizontally, then vertically, without changing the 3-D mask."""
    x = np.asarray(values, dtype=np.float32)
    valid = np.asarray(observation_mask, dtype=bool)
    if x.ndim != 4 or valid.shape != x.shape:
        raise ValueError(f"expected equal [T,D,H,W] arrays, got {x.shape}, {valid.shape}")
    if np.any(~np.isfinite(x[valid])):
        raise ValueError("observed values must be finite")
    result = np.where(valid, x, np.nan).astype(np.float32)
    available = valid.copy()
    for _ in range(4):
        prior, prior_available = result.copy(), available.copy()
        changed = np.zeros_like(available)
        for iy in range(x.shape[2]):
            for ix in range(x.shape[3]):
                ys, ye = max(0, iy - 1), min(x.shape[2], iy + 2)
                xs, xe = max(0, ix - 1), min(x.shape[3], ix + 2)
                neighbours = prior[:, :, ys:ye, xs:xe]
                ok = prior_available[:, :, ys:ye, xs:xe]
                count = ok.sum(axis=(2, 3))
                fill = np.divide(
                    np.where(ok, neighbours, 0.0).sum(axis=(2, 3)), count,
                    out=np.zeros_like(count, dtype=np.float32), where=count > 0,
                )
                take = (~available[:, :, iy, ix]) & (count > 0)
                result[:, :, iy, ix][take] = fill[take]
                changed[:, :, iy, ix] = take
        available |= changed
        if not changed.any():
            break
    sums = np.where(valid, x, 0.0).sum(axis=(2, 3), keepdims=True)
    counts = valid.sum(axis=(2, 3), keepdims=True)
    fallback = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    result = np.where(available, result, fallback).astype(np.float32)
    for step in range(result.shape[0]):
        observed = np.flatnonzero(counts[step, :, 0, 0] > 0)
        if observed.size == 0:
            continue
        for level in np.flatnonzero(counts[step, :, 0, 0] == 0):
            shallower, deeper = observed[observed < level], observed[observed > level]
            if shallower.size and deeper.size:
                upper, lower = int(shallower[-1]), int(deeper[0])
                weight = (level - upper) / (lower - upper)
                result[step, level] = (1.0 - weight) * result[step, upper] + weight * result[step, lower]
            elif shallower.size:
                result[step, level] = result[step, int(shallower[-1])]
            else:
                result[step, level] = result[step, int(deeper[0])]
    return result, valid.mean(axis=1, dtype=np.float32)
