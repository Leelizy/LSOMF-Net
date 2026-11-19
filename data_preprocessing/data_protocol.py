"""Mask generation, observed-only filling and data splits."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


PREPROCESSING_VERSION = "lsomf-safe-fill-v6-dual-prior-ocgc-v2-fixed-mask"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return a reproducible content hash for the input data file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate_seed(base_seed: int, rate: float) -> int:
    """Derive independent, reproducible mask streams for different rates."""

    rate_key = int(round(float(rate) * 1_000_000.0))
    sequence = np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF, rate_key])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def generate_masks_timestep(
    total: int,
    height: int,
    width: int,
    rate: float,
    base_seed: int = 42,
) -> np.ndarray:
    """Generate masks (1=observed, 0=missing) with a separate stream per rate."""

    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"mask rate must be in [0, 1], got {rate}")
    if total < 1 or height < 1 or width < 1:
        raise ValueError("mask dimensions must be positive")

    rng = np.random.default_rng(_rate_seed(int(base_seed), rate))
    masks = (rng.random((total, height, width)) > rate).astype(np.float32)
    for t in range(total):
        if masks[t].sum() == 0:
            masks[t, rng.integers(0, height), rng.integers(0, width)] = 1.0
    return masks


def neighbor_fill_single(
    field: np.ndarray,
    mask: np.ndarray,
    iters: int = 4,
) -> np.ndarray:
    """Fill a frame from observed cells without using hidden target values."""

    field = np.asarray(field)
    mask = np.asarray(mask)
    if field.ndim != 3 or mask.ndim != 2 or field.shape[1:] != mask.shape:
        raise ValueError(
            f"field must be [C,H,W] and mask [H,W], got {field.shape} and {mask.shape}"
        )

    observed = mask > 0.5
    filled = np.zeros_like(field)
    if observed.any():
        filled[:, observed] = field[:, observed]
        fallback = field[:, observed].mean(axis=1)
    else:
        fallback = np.zeros(field.shape[0], dtype=field.dtype)

    current = observed.copy()
    for _ in range(max(0, int(iters))):
        new_filled = filled.copy()
        added = np.zeros_like(current)
        for h in range(field.shape[1]):
            for w in range(field.shape[2]):
                if current[h, w]:
                    continue
                hs, he = max(0, h - 1), min(field.shape[1], h + 2)
                ws, we = max(0, w - 1), min(field.shape[2], w + 2)
                valid = current[hs:he, ws:we]
                if valid.any():
                    values = filled[:, hs:he, ws:we][:, valid]
                    new_filled[:, h, w] = values.mean(axis=1)
                    added[h, w] = True
        filled = new_filled
        current |= added
        if not added.any():
            break

    # Unfilled cells use the observed mean, or zero when nothing is observed.
    if (~current).any():
        filled[:, ~current] = fallback[:, None]
    return filled


def neighbor_fill_per_timestep(
    fields: np.ndarray,
    masks: np.ndarray,
    iters: int = 4,
) -> np.ndarray:
    """Apply :func:`neighbor_fill_single` independently to every time step."""

    fields = np.asarray(fields)
    masks = np.asarray(masks)
    if fields.ndim != 4 or masks.ndim != 3 or fields.shape[0] != masks.shape[0]:
        raise ValueError(
            f"fields must be [T,C,H,W] and masks [T,H,W], got {fields.shape} and {masks.shape}"
        )
    return np.stack(
        [neighbor_fill_single(field, mask, iters=iters) for field, mask in zip(fields, masks)]
    )


def split_window_indices(
    total_frames: int,
    t_in: int,
    h_out: int,
    train_end: int,
    val_end: int,
) -> dict[str, np.ndarray]:
    """Split by target timestamps; exclude targets crossing a boundary."""

    if not 0 < train_end < val_end <= total_frames:
        raise ValueError(
            "require 0 < train_end < val_end <= total_frames, got "
            f"{train_end}, {val_end}, {total_frames}"
        )
    n_windows = total_frames - t_in - h_out + 1
    if n_windows < 1:
        raise ValueError("not enough frames for one input/target window")

    starts = np.arange(n_windows, dtype=np.int64)
    target_start = starts + int(t_in)
    target_end = target_start + int(h_out)
    return {
        "idx_train": starts[target_end <= train_end],
        "idx_val": starts[(target_start >= train_end) & (target_end <= val_end)],
        "idx_test": starts[target_start >= val_end],
    }
