"""Prepare the supplied regional data for the main experiment."""
from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA

from config import ANCHOR_IDX, CACHE_DIR, CACHE_FILE, GH, GW, H_OUT, MASK_SEED, N_DEP, N_ST, TRAIN_END, T_IN, VAL_END
from data_preprocessing.data_protocol import PREPROCESSING_VERSION, generate_masks_timestep, neighbor_fill_per_timestep, sha256_file, split_window_indices
from data_preprocessing.argo import initialize_argo

DEFAULT_DATA_PATH = CACHE_DIR / CACHE_FILE
N_PCA = N_ST - 1
ARGO_PREPROCESSING_VERSION = "lsomf-argo-mask-v1-gdcsm-neighbor-vertical-fill"

def prepare_data(
    mask_ratio: float = 0.0,
    data_path: Path = DEFAULT_DATA_PATH,
    mask_seed: int = MASK_SEED,
    anchor_idx: int = ANCHOR_IDX,
    train_end: int = TRAIN_END,
    val_end: int = VAL_END,
    prior_mode: str = "full",
    argo_mask_path: Path | None = None,
) -> dict[str, object]:
    """Prepare masked inputs and split samples by target timestamp."""

    if prior_mode not in {"full", "none"}:
        raise ValueError(f"prior_mode must be 'full' or 'none', got {prior_mode!r}")
    data_path = Path(data_path)
    anchor_idx = int(anchor_idx)
    train_end = int(train_end)
    val_end = int(val_end)
    if not 0 <= anchor_idx < N_DEP:
        raise ValueError(f"anchor_idx must be in [0, {N_DEP - 1}], got {anchor_idx}")
    with np.load(data_path) as data:
        if "svel" not in data:
            raise KeyError(f"{data_path} does not contain an 'svel' array")
        raw = data["svel"][:, :, :GH, :GW].astype(np.float32)
        coordinates = {key: np.asarray(data[key]) for key in ("pres", "lat", "lon") if key in data}
        input_time = np.asarray(data["time"]) if "time" in data else None
    if not np.isfinite(raw).all():
        raise ValueError("Input sound-speed field contains NaN or Inf")
    if raw.shape[1] != N_DEP or raw.shape[2:] != (GH, GW):
        raise ValueError(
            f"Expected cropped field [T,{N_DEP},{GH},{GW}], got {raw.shape}"
        )
    total = len(raw)
    splits = split_window_indices(total, T_IN, H_OUT, train_end, val_end)
    train_indices, val_indices, test_indices = splits["idx_train"], splits["idx_val"], splits["idx_test"]
    if not len(train_indices) or not len(val_indices) or not len(test_indices):
        raise ValueError(
            "Target-time split requires non-empty train, validation and test partitions: "
            f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
        )
    expected_months = np.datetime64("2004-01", "M") + np.arange(total)
    if input_time is None:
        warnings.warn("Input NPZ has no 'time'; assuming consecutive months starting 2004-01", stacklevel=2)
    else:
        if input_time.ndim != 1 or len(input_time) != total or input_time.dtype.kind not in "MUS":
            raise ValueError("NPZ time must contain one datetime or ISO date string per frame")
        try:
            input_months = input_time.astype("datetime64[M]")
        except ValueError as exc:
            raise ValueError("NPZ time must contain valid monthly dates") from exc
        if not np.array_equal(input_months, expected_months):
            raise ValueError("NPZ time must be consecutive months starting 2004-01")

    print(f"  Preparing {total} frames; input mode={'Argo mask' if argo_mask_path else 'synthetic mask'}")
    start = time.time()
    argo_mask_hash = None
    if argo_mask_path is not None:
        if mask_ratio != 0:
            raise ValueError("an Argo mask cannot be combined with a synthetic mask ratio")
        with np.load(Path(argo_mask_path)) as argo:
            mask_values = argo["observation_mask"]
            if mask_values.shape != raw.shape or not np.isin(mask_values, [0, 1]).all():
                raise ValueError("Argo observation_mask must be binary and match GDCSM [T,58,16,16]")
            if "bin_start_utc" in argo:
                argo_months = np.asarray(argo["bin_start_utc"]).astype("datetime64[M]")
                if not np.array_equal(argo_months, expected_months):
                    raise ValueError("Argo mask time bins must match consecutive months starting 2004-01")
            for data_key, argo_key in (("pres", "pressure_dbar"), ("lat", "latitude"), ("lon", "longitude_360")):
                if data_key in coordinates and argo_key in argo:
                    left, right = coordinates[data_key], np.asarray(argo[argo_key])
                    if data_key == "lon":
                        left, right = left % 360, right % 360
                    if left.shape != right.shape or not np.allclose(left, right, rtol=0, atol=1e-4):
                        raise ValueError(f"Argo {argo_key} does not match GDCSM {data_key}")
            direct = np.asarray(mask_values, dtype=bool)
        argo_mask_hash = hashlib.sha256(direct.tobytes(order="C")).hexdigest()
        filled, masks = initialize_argo(raw, direct)
        observed_raw = np.where(direct, raw, 0.0).astype(np.float32)
        direct_observation_mask = direct.astype(np.float32)
        mask_ratio = 1.0 - float(masks.mean())
        print(f"  Argo 3-D missing ratio: {mask_ratio:.6f}")
    else:
        masks = generate_masks_timestep(total, GH, GW, mask_ratio, base_seed=int(mask_seed))
        observed_raw = np.where(masks[:, None] > 0.5, raw, 0.0).astype(np.float32)
        direct_observation_mask = np.broadcast_to(masks[:, None], raw.shape).astype(np.float32, copy=True)
        if mask_ratio > 0:
            kept = masks.mean() * GH * GW
            print(f"  Mean observed cells: {kept:.1f}/{GH * GW}")
            filled = neighbor_fill_per_timestep(raw, masks, iters=4)
            print(f"  Neighbor filling finished in {time.time() - start:.1f}s")
        else:
            filled = raw.copy()
    # Keep only observed values for reconstruction consistency checks.

    # Fit PCA using complete training months only.
    anchor_train = raw[:train_end, anchor_idx : anchor_idx + 1]
    residual_train = raw[:train_end] - anchor_train
    train_flat = residual_train.transpose(0, 2, 3, 1).reshape(-1, N_DEP)
    pca = PCA(n_components=N_PCA, svd_solver="covariance_eigh").fit(train_flat)
    print(f"  ACPE PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    def encode(fields: np.ndarray) -> np.ndarray:
        anchor = fields[:, anchor_idx : anchor_idx + 1]
        residual = fields - anchor
        flat = residual.transpose(0, 2, 3, 1).reshape(-1, N_DEP)
        coeff = (flat - pca.mean_) @ pca.components_.T
        coeff = coeff.reshape(len(fields), GH, GW, N_PCA).transpose(0, 3, 1, 2)
        return np.concatenate([anchor, coeff], axis=1).astype(np.float32)

    true_encoded = encode(raw)
    noisy_encoded = encode(filled)

    # Compute climatology from training months only.
    calendar_index_np = np.arange(total, dtype=np.int64) % 12
    month_index_np = (
        calendar_index_np
        if prior_mode == "full"
        else np.zeros(total, dtype=np.int64)
    )
    seasonal_raw = np.stack(
        [
            raw[:train_end][calendar_index_np[:train_end] == month].mean(axis=0)
            for month in range(12)
        ]
    ).astype(np.float32)
    seasonal_encoded = encode(seasonal_raw)

    # Fit normalization using training months only.
    mean = true_encoded[:train_end].mean(axis=(0, 2, 3)).astype(np.float32)
    scale = true_encoded[:train_end].std(axis=(0, 2, 3)).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    noisy_norm = (noisy_encoded - mean[None, :, None, None]) / scale[None, :, None, None]
    true_norm = (true_encoded - mean[None, :, None, None]) / scale[None, :, None, None]
    seasonal_norm = (seasonal_encoded - mean[None, :, None, None]) / scale[None, :, None, None]

    noisy_tensor = torch.tensor(noisy_norm, dtype=torch.float32)
    true_tensor = torch.tensor(true_norm, dtype=torch.float32)
    seasonal_tensor = torch.tensor(seasonal_norm, dtype=torch.float32)
    month_index = torch.from_numpy(month_index_np)

    xs, months, ys, true_inputs, window_masks, window_direct_masks, baseline_inputs = [], [], [], [], [], [], []
    for i in range(total - T_IN - H_OUT + 1):
        xs.append(noisy_tensor[i : i + T_IN])
        months.append(month_index[i + T_IN - 1])
        ys.append(true_tensor[i + T_IN : i + T_IN + H_OUT])
        true_inputs.append(true_tensor[i : i + T_IN])
        window_masks.append(torch.from_numpy(masks[i : i + T_IN]))
        window_direct_masks.append(
            torch.from_numpy(direct_observation_mask[i : i + T_IN])
        )
        if prior_mode == "full":
            baseline_inputs.append(seasonal_tensor[calendar_index_np[i : i + T_IN]])
        else:
            baseline_inputs.append(noisy_tensor[i : i + T_IN])

    xs = torch.stack(xs)
    months = torch.stack(months)
    ys = torch.stack(ys)
    true_inputs = torch.stack(true_inputs)
    window_masks = torch.stack(window_masks)
    window_direct_masks = torch.stack(window_direct_masks)
    baseline_inputs = torch.stack(baseline_inputs)

    print(
        f"  Target-time windows: total={len(xs)}, train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}"
    )

    data_hash = sha256_file(data_path)
    metadata = {
        "preprocessing_version": ARGO_PREPROCESSING_VERSION if argo_mask_path else PREPROCESSING_VERSION,
        "data_path": str(data_path.resolve()),
        "data_sha256": data_hash,
        "mask_ratio": float(mask_ratio),
        "mask_seed": int(mask_seed),
        "anchor_idx": anchor_idx,
        "n_pca": int(N_PCA),
        "t_in": int(T_IN),
        "h_out": int(H_OUT),
        "month_anchor": "last_input_frame",
        "reconstruction_truth": "raw_58_depth",
        "reconstruction_baseline": "training_monthly_climatology" if prior_mode == "full" else "filled_sparse_input",
        "reconstruction_metric": "pooled_rmse_over_all_window_depth_cells",
        "reconstruction_observed_consistency": "direct_time_depth_cell_mask_after_decode",
        "forecast_baseline_protocol": "raw-field-baseline",
        "forecast_input_representation": "acpe_of_observed_only_filled_raw_58_plus_explicit_mask",
        "physical_reconstruction_mask": "direct_time_depth_cell_observation_mask",
        "sparse_input_mode": "gdcsm_at_raw_argo_mask" if argo_mask_path else "synthetic_mask",
        "sparse_input_value_source": "GDCSM at Argo-observed locations" if argo_mask_path else "complete GDCSM / synthetic-mask protocol",
        "argo_mask_sha256": argo_mask_hash,
        "prior_mode": prior_mode,
        "pca_solver": "covariance_eigh",
        "scikit_learn_version": sklearn.__version__,
        "calendar_start_assumption": "2004-01",
        "calendar_time_verified": input_time is not None,
        "raw_shape": [int(v) for v in raw.shape],
        "train_end": train_end,
        "val_end": val_end,
        "split": "target_timestamp",
        "window_counts": {
            "train": int(len(train_indices)),
            "val": int(len(val_indices)),
            "test": int(len(test_indices)),
        },
    }

    return {
        "xs": xs,
        "ms": months,
        "ys": ys,
        "xt": true_inputs,
        "xb": baseline_inputs,
        "mw": window_masks,
        "md": window_direct_masks,
        "idx_train": train_indices,
        "idx_tr": train_indices,
        "idx_val": val_indices,
        "idx_test": test_indices,
        "svel_raw": torch.from_numpy(raw),
        "svel_observed": torch.from_numpy(observed_raw),
        "seasonal_raw": torch.from_numpy(seasonal_raw),
        "pca_comps": torch.from_numpy(pca.components_.copy()).float(),
        "pca_mean": torch.from_numpy(pca.mean_.copy()).float(),
        "mu_t": torch.from_numpy(mean.copy()).float(),
        "sig_t": torch.from_numpy(scale.copy()).float(),
        "mask_ratio": mask_ratio,
        "mask_seed": int(mask_seed),
        "anchor_idx": anchor_idx,
        "metadata": metadata,
    }


def decode_to_raw(z_state: torch.Tensor, data: dict[str, object]) -> torch.Tensor:
    """Decode a normalized nine-channel state back to the 58-depth field."""

    mean = data["mu_t"].view(1, -1, 1, 1).to(z_state.device)
    scale = data["sig_t"].view(1, -1, 1, 1).to(z_state.device)
    components = data["pca_comps"].to(z_state.device)
    pca_mean = data["pca_mean"].to(z_state.device)
    state = z_state * scale + mean
    anchor = state[:, 0:1]
    coefficients = state[:, 1 : 1 + N_PCA]
    batch, _, height, width = coefficients.shape
    coeff_flat = coefficients.permute(0, 2, 3, 1).reshape(-1, N_PCA)
    residual = coeff_flat @ components + pca_mean.unsqueeze(0)
    residual = residual.reshape(batch, height, width, N_DEP).permute(0, 3, 1, 2)
    return anchor + residual


def split_indices(data: dict[str, object], split: str) -> np.ndarray:
    """Return a named split while keeping the optional test split explicit."""

    key = {"train": "idx_train", "val": "idx_val", "test": "idx_test"}.get(split)
    if key is None:
        raise ValueError(f"split must be train, val, or test; got {split!r}")
    indices = np.asarray(data[key], dtype=np.int64)
    if not len(indices):
        raise ValueError(f"Requested {split} split is empty")
    return indices
