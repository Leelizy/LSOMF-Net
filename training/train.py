"""Train and evaluate LSOMF-Net with two-stage optimization."""
from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    ANCHOR_IDX, BATCH, CODEC_EPOCHS, DEVICE, FT_EPOCHS, GH, GW, H_OUT,
    LAMBDA_OBS, LAMBDA_RAW_REC, LAMBDA_REC, LR_CODEC, LR_FT, MASK_SEED,
    N_DEP, OUT_DIR, TRAIN_END, T_IN, VAL_END,
)
from data_preprocessing.data_protocol import PREPROCESSING_VERSION
from data_preprocessing.prepare import ARGO_PREPROCESSING_VERSION, DEFAULT_DATA_PATH, decode_to_raw, prepare_data, split_indices
from evaluation.metrics import evaluate_final
from model.lsomf_net import MODEL_VERSION, get_model, param_table

CKPT_DIR = OUT_DIR / "best_checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def physical_rmse(z_state: torch.Tensor, target_raw: torch.Tensor, data: dict) -> torch.Tensor:
    return torch.sqrt(torch.mean((decode_to_raw(z_state, data) - target_raw) ** 2) + 1e-8)


def decode_sequence_to_raw(z_state: torch.Tensor, data: dict) -> torch.Tensor:
    """Decode a normalized latent sequence to ``[B,T,58,H,W]``."""

    if z_state.ndim != 5:
        raise ValueError(f"expected latent sequence [B,T,C,H,W], got {tuple(z_state.shape)}")
    batch, steps, channels, height, width = z_state.shape
    decoded = decode_to_raw(z_state.reshape(-1, channels, height, width), data)
    return decoded.reshape(batch, steps, N_DEP, GH, GW)


def masked_physical_reconstruction_loss(
    refined: torch.Tensor,
    truth_raw: torch.Tensor,
    mask: torch.Tensor,
    data: dict,
) -> torch.Tensor:
    """Compute reconstruction loss on unobserved cells in physical space."""

    prediction_raw = decode_sequence_to_raw(refined, data)
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    if mask.shape != prediction_raw.shape:
        raise ValueError(
            "physical reconstruction mask must match [B,T,D,H,W], got "
            f"{tuple(mask.shape)} for prediction {tuple(prediction_raw.shape)}"
        )
    missing = 1.0 - mask
    difference = prediction_raw - truth_raw
    missing_count = missing.sum()
    if missing_count <= 0:
        return difference.new_zeros(())
    return ((missing * difference) ** 2).sum() / missing_count


def masked_reconstruction_loss(
    refined: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    lambda_obs: float = LAMBDA_OBS,
) -> torch.Tensor:
    """Weight missing and observed cells; normalize by spatial cell count."""

    observed = mask.unsqueeze(2)
    difference = refined - truth
    missing = 1.0 - observed
    missing_count, observed_count = missing.sum(), observed.sum()
    missing_term = (
        ((missing * difference) ** 2).sum() / missing_count
        if missing_count > 0
        else difference.new_zeros(())
    )
    observed_term = (
        ((observed * difference) ** 2).sum() / observed_count
        if observed_count > 0
        else difference.new_zeros(())
    )
    return missing_term + lambda_obs * observed_term


@torch.no_grad()
def reconstruction_rmse(
    model: nn.Module,
    data: dict[str, object],
    indices: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute reconstruction RMSE pooled across windows."""

    model.eval()
    device = next(model.parameters()).device
    indices = data["idx_val"] if indices is None else indices
    missing_sse = 0.0
    missing_count = 0
    observed_sse = 0.0
    observed_count = 0
    full_sse = 0.0
    full_count = 0
    for index in np.asarray(indices, dtype=np.int64):
        index = int(index)
        refined, _ = model(
            data["xs"][index : index + 1].to(device),
            data["ms"][index : index + 1].to(device),
            mask_ratio=data["mask_ratio"],
            spatial_mask=data["mw"][index : index + 1].to(device),
            baseline=data["xb"][index : index + 1].to(device),
        )
        _, channels, height, width = refined[0].shape
        refined_raw = decode_to_raw(refined[0].reshape(-1, channels, height, width), data)
        truth_raw = data["svel_raw"][index : index + T_IN].to(device)
        direct_mask = data["md"][index].to(device)
        observed_raw = data["svel_observed"][index : index + T_IN].to(device)
        refined_raw = torch.where(direct_mask > 0.5, observed_raw, refined_raw)
        difference = (refined_raw - truth_raw) ** 2
        missing_selection = (direct_mask <= 0.5).bool()
        observed_selection = (direct_mask > 0.5).bool()
        missing_sse += float(difference[missing_selection].sum().item())
        missing_count += int(missing_selection.sum().item())
        observed_sse += float(difference[observed_selection].sum().item())
        observed_count += int(observed_selection.sum().item())
        full_sse += float(difference.sum().item())
        full_count += int(difference.numel())

    def pooled_rmse(sum_squared_error: float, count: int) -> float | None:
        if count == 0:
            return None
        return float(np.sqrt(sum_squared_error / count))

    return {
        "recon_miss_rmse": pooled_rmse(missing_sse, missing_count),
        "recon_obs_rmse": pooled_rmse(observed_sse, observed_count),
        "recon_full_rmse": pooled_rmse(full_sse, full_count),
    }


@torch.no_grad()
def quick_raw_eval(
    model: nn.Module,
    data: dict[str, object],
    indices: np.ndarray | None = None,
) -> float:
    """Compute raw forecast RMSE for model selection or final reporting."""

    model.eval()
    device = next(model.parameters()).device
    indices = data["idx_val"] if indices is None else indices
    errors = []
    for index in np.asarray(indices, dtype=np.int64):
        index = int(index)
        _, forecast = model(
            data["xs"][index : index + 1].to(device),
            data["ms"][index : index + 1].to(device),
            mask_ratio=data["mask_ratio"],
            spatial_mask=data["mw"][index : index + 1].to(device),
            baseline=data["xb"][index : index + 1].to(device),
        )
        prediction = decode_to_raw(forecast[:, 0], data)
        truth = data["svel_raw"][index + T_IN].unsqueeze(0).to(device)
        errors.append(torch.sqrt(torch.mean((prediction - truth) ** 2)).item())
    return float(np.mean(errors))


def train(
    model: nn.Module,
    data: dict[str, object],
    codec_epochs: int = CODEC_EPOCHS,
    ft_epochs: int = FT_EPOCHS,
    lr_codec: float = LR_CODEC,
    lr_ft: float = LR_FT,
    ft_patience: int = 6,
    disable_early_stop: bool = False,
    selection_metric: str = "forecast",
    resume: bool = False,
) -> float:
    """Train in two stages and select a checkpoint using validation metrics."""

    if selection_metric not in {"forecast", "reconstruction"}:
        raise ValueError(
            f"selection_metric must be 'forecast' or 'reconstruction', got {selection_metric!r}"
        )

    train_indices = split_indices(data, "train")
    val_indices = split_indices(data, "val")
    xs_train = data["xs"][train_indices]
    months_train = data["ms"][train_indices]
    targets_train = data["ys"][train_indices]
    true_inputs = data["xt"][train_indices]
    masks_train = data["mw"][train_indices]
    direct_masks_train = data["md"][train_indices]
    baselines_train = data["xb"][train_indices]
    raw = data["svel_raw"]
    mask_ratio = data["mask_ratio"]
    device = next(model.parameters()).device

    criterion = nn.MSELoss()
    if lr_codec <= 0 or lr_ft <= 0:
        raise ValueError(f"learning rates must be positive, got codec={lr_codec}, ft={lr_ft}")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr_codec), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6
    )
    best_raw = float("inf")
    best_selection = float("inf")
    selection_metric_name = (
        "validation_missing_reconstruction_rmse"
        if selection_metric == "reconstruction"
        else "validation_forecast_rmse"
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_codec_state: dict[str, torch.Tensor] | None = None
    best_codec = float("inf")
    codec_no_improvement = 0

    def clone_state() -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    def train_one_epoch(in_finetune: bool) -> None:
        model.train()
        permutation = np.random.permutation(len(xs_train))
        for start in range(0, len(xs_train), BATCH):
            batch_indices = permutation[start : start + BATCH]
            refined, forecast = model(
                xs_train[batch_indices].to(device),
                months_train[batch_indices].to(device),
                mask_ratio=mask_ratio,
                spatial_mask=masks_train[batch_indices].to(device),
                baseline=baselines_train[batch_indices].to(device),
            )
            if not in_finetune:
                forecast_loss = torch.sqrt(
                    criterion(
                        forecast[:, 0],
                        targets_train[batch_indices][:, 0].to(device),
                    )
                    + 1e-8
                )
                reconstruction_loss = masked_reconstruction_loss(
                    refined,
                    true_inputs[batch_indices].to(device),
                    masks_train[batch_indices].to(device),
                )
                loss = forecast_loss + LAMBDA_REC * reconstruction_loss
            else:
                raw_targets = torch.stack(
                    [raw[int(train_indices[index]) + T_IN] for index in batch_indices]
                ).to(device)
                input_raw_targets = torch.stack(
                    [
                        raw[int(train_indices[index]) : int(train_indices[index]) + T_IN]
                        for index in batch_indices
                    ]
                ).to(device)
                raw_reconstruction_loss = masked_physical_reconstruction_loss(
                    refined,
                    input_raw_targets,
                    direct_masks_train[batch_indices].to(device),
                    data,
                )
                loss = physical_rmse(forecast[:, 0], raw_targets, data)
                loss = loss + LAMBDA_RAW_REC * raw_reconstruction_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    @torch.no_grad()
    def validation_codec_rmse() -> float:
        model.eval()
        _, validation_forecast = model(
            data["xs"][val_indices].to(device),
            data["ms"][val_indices].to(device),
            mask_ratio=mask_ratio,
            spatial_mask=data["mw"][val_indices].to(device),
            baseline=data["xb"][val_indices].to(device),
        )
        return float(
            torch.sqrt(
                criterion(
                    validation_forecast[:, 0],
                    data["ys"][val_indices][:, 0].to(device),
                )
                + 1e-8
            ).item()
        )

    def observe(stage: str, epoch: int, validation_codec: float) -> tuple[float, float]:
        nonlocal best_raw, best_selection, best_state
        raw_score = quick_raw_eval(model, data, val_indices)
        reconstruction = reconstruction_rmse(model, data, val_indices)
        missing_score = reconstruction["recon_miss_rmse"]
        if selection_metric == "reconstruction" and missing_score is not None:
            selection_score = missing_score
        else:
            selection_score = raw_score
        if raw_score < best_raw:
            best_raw = raw_score
        if selection_score < best_selection:
            best_selection = selection_score
            best_state = clone_state()
        missing_text = "n/a" if missing_score is None else f"{missing_score:.4f}"
        print(
            f"    epoch {epoch:3d} [{stage}] codec={validation_codec:.4f} "
            f"val_forecast={raw_score:.4f} val_recon_missing={missing_text} "
            f"best_{selection_metric_name}={best_selection:.4f}"
        )
        return raw_score, selection_score

    # Codec early stopping still proceeds to physical fine-tuning.
    for epoch in range(1, max(0, int(codec_epochs)) + 1):
        train_one_epoch(in_finetune=False)
        validation_codec = validation_codec_rmse()
        scheduler.step(validation_codec)
        if validation_codec < best_codec:
            best_codec = validation_codec
            best_codec_state = clone_state()
            codec_no_improvement = 0
        else:
            codec_no_improvement += 1
        if epoch % 5 == 0 or epoch == codec_epochs:
            observe("ACPE/codec", epoch, validation_codec)
        if (not disable_early_stop) and codec_no_improvement >= 15 and epoch >= 20:
            print("  Early stopping: codec stage plateaued; continuing to physical fine-tuning")
            break

    if ft_epochs > 0:
        if not resume and best_state is not None:
            model.load_state_dict(best_state)
            print("  Restored best validation-selection checkpoint before physical fine-tuning")
        elif not resume and best_codec_state is not None:
            model.load_state_dict(best_codec_state)
            print("  Restored best codec checkpoint before physical fine-tuning")
        # Restart checkpoint selection from the restored fine-tuning state.
        best_state = clone_state()
        best_selection = float("inf")
        _, _ = observe("FT-start", 0, validation_codec_rmse())
        for group in optimizer.param_groups:
            group["lr"] = float(lr_ft)

    ft_plateau = 0
    for epoch in range(1, max(0, int(ft_epochs)) + 1):
        train_one_epoch(in_finetune=True)
        validation_codec = validation_codec_rmse()
        previous_best = best_selection
        _, selection_score = observe("FT", epoch, validation_codec)
        ft_plateau = 0 if selection_score < previous_best else ft_plateau + 1
        if (not disable_early_stop) and ft_plateau >= ft_patience:
            print("  Early stopping: physical validation has plateaued")
            break

    if best_state is None:
        best_raw = quick_raw_eval(model, data, val_indices)
        best_state = clone_state()
    model.load_state_dict(best_state)
    print(
        f"  Best validation forecast RMSE: {best_raw:.4f}; "
        f"selection={selection_metric_name}:{best_selection:.4f}"
    )
    return best_raw



def _load_checkpoint_payload(path: Path, device: str | torch.device) -> dict[str, object]:
    """Load a checkpoint and verify its version."""

    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError(
            "Checkpoint metadata are required to verify preprocessing and "
            "evaluation compatibility."
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Checkpoint has no preprocessing metadata")
    if metadata.get("preprocessing_version") not in {PREPROCESSING_VERSION, ARGO_PREPROCESSING_VERSION}:
        raise RuntimeError("Checkpoint was created with a different preprocessing version")
    if metadata.get("model_version") != MODEL_VERSION:
        raise RuntimeError("Checkpoint was created with a different model version")
    return payload


def _set_all_seeds(seed: int) -> None:
    """Seed initialization, NumPy window ordering, and Python-side sampling."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final LSOMF-Net model")
    parser.add_argument("--epochs", type=int, default=CODEC_EPOCHS)
    parser.add_argument("--ft", type=int, default=FT_EPOCHS)
    parser.add_argument(
        "--lr-codec",
        type=float,
        default=LR_CODEC,
        help=f"Codec-stage Adam learning rate (default: {LR_CODEC:g}).",
    )
    parser.add_argument(
        "--lr-ft",
        type=float,
        default=LR_FT,
        help=f"Physical fine-tuning Adam learning rate (default: {LR_FT:g}).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mask", type=float, default=0.0)
    parser.add_argument(
        "--anchor-idx",
        type=int,
        default=ANCHOR_IDX,
        help="Anchor depth index; default 21 (220 dbar in the supplied cache).",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=MASK_SEED,
        help="Base seed for masks; fixed across model seeds for fair comparison.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--argo-mask", type=Path, default=None, help="Argo 3-D observation mask NPZ aligned with the GDCSM data")
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help="Use the experiment configuration with up to 40+50 epochs.",
    )
    parser.add_argument("--train-end", type=int, default=TRAIN_END)
    parser.add_argument("--val-end", type=int, default=VAL_END)
    parser.add_argument(
        "--prior-mode",
        choices=("full", "none"),
        default="full",
        help=(
            "full uses training-month climatology and calendar phase; none uses "
            "filled sparse inputs as the baseline and fixes month indices at zero."
        ),
    )
    parser.add_argument("--checkpoint-tag", default="")
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="test",
        help="Use the common held-out test split by default; val is for diagnostics.",
    )
    parser.add_argument("--result-path", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Load a checkpoint and run only the fine-tuning stage.",
    )
    parser.add_argument("--ft-patience", type=int, default=6)
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="Run all requested codec and physical-fine-tuning epochs without early stopping.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("forecast", "reconstruction"),
        default="forecast",
        help=(
            "Validation metric used to select the final checkpoint. The raw "
            "forecast metric is the fair default for forecast baselines."
        ),
    )
    args = parser.parse_args()
    if args.argo_mask is not None and args.mask != 0:
        parser.error("--argo-mask cannot be combined with a synthetic --mask")

    if args.paper_protocol:
        args.epochs = CODEC_EPOCHS
        args.ft = FT_EPOCHS
        args.lr_codec = LR_CODEC
        args.lr_ft = LR_FT
        args.prior_mode = "full"
        args.selection_metric = "forecast"

    print("\n" + "=" * 72)
    if args.seed is None:
        args.seed = secrets.randbits(32)
    print(
        f"  LSOMF-Net: OCGC + CA-OMF, "
        f"codec={args.epochs} epochs, ft={args.ft} epochs, "
        f"seed={args.seed}, mask={args.mask}"
    )
    print(f"  Learning rates: codec={args.lr_codec:g}, fine-tuning={args.lr_ft:g}")
    if args.paper_protocol:
        print("  Experiment preset: up to 40+50 epochs")
    print("=" * 72)
    _set_all_seeds(args.seed)

    mask_seed = int(args.mask_seed)

    data = prepare_data(
        mask_ratio=args.mask,
        data_path=args.data_path,
        mask_seed=mask_seed,
        anchor_idx=args.anchor_idx,
        train_end=args.train_end,
        val_end=args.val_end,
        prior_mode=args.prior_mode,
        argo_mask_path=args.argo_mask,
    )
    # Initialize the model after setting its seed.
    model = get_model().to(DEVICE)
    print(param_table())

    if args.resume is not None:
        payload = _load_checkpoint_payload(args.resume, DEVICE)
        checkpoint_meta = payload["metadata"]
        for key in (
            "preprocessing_version",
            "data_sha256",
            "mask_ratio",
            "mask_seed",
            "anchor_idx",
            "t_in",
            "h_out",
            "train_end",
            "val_end",
            "prior_mode",
            "sparse_input_mode",
            "argo_mask_sha256",
        ):
            saved_value = checkpoint_meta.get(key, "synthetic_mask") if key == "sparse_input_mode" else checkpoint_meta.get(key)
            if saved_value != data["metadata"].get(key):
                raise RuntimeError(
                    f"Resume checkpoint metadata mismatch for {key}: "
                    f"checkpoint={saved_value!r}, current={data['metadata'].get(key)!r}"
                )
        for key in ("pca_solver", "scikit_learn_version"):
            if key in checkpoint_meta and checkpoint_meta[key] != data["metadata"][key]:
                raise RuntimeError(f"Resume checkpoint metadata mismatch for {key}")
        model.load_state_dict(payload["model_state"])
        print(f"  Resumed checkpoint: {args.resume}")
        codec_epochs, ft_epochs = 0, args.ft
        resume = True
    else:
        codec_epochs, ft_epochs = args.epochs, args.ft
        resume = False

    start = time.time()
    train(
        model,
        data,
        codec_epochs=codec_epochs,
        ft_epochs=ft_epochs,
        lr_codec=args.lr_codec,
        lr_ft=args.lr_ft,
        ft_patience=args.ft_patience,
        disable_early_stop=args.disable_early_stop,
        selection_metric=args.selection_metric,
        resume=resume,
    )
    print(f"  Training time: {time.time() - start:.0f}s")

    results = evaluate_final(model, data, eval_split=args.eval_split)
    eval_indices = split_indices(data, args.eval_split)
    reconstruction = reconstruction_rmse(model, data, eval_indices)
    missing_rmse = reconstruction["recon_miss_rmse"]
    missing_text = "n/a" if missing_rmse is None else f"{missing_rmse:.4f}"
    print("\n" + "=" * 72)
    print(f"  Lead-1 physical RMSE: {results['lead1']:.4f}")
    print(f"  All-lead physical RMSE: {results['all_lead']:.4f}")
    print(f"  Per-lead: {[f'{v:.3f}' for v in results['per_lead']]}")
    print(
        f"  Quadrants: { {k: f'{v:.3f}' for k, v in results['quad'].items()} } "
        f"gap={results['quad_gap']:.3f}"
    )
    print(
        f"  Reconstruction RMSE: missing={missing_text}, "
        f"observed={reconstruction['recon_obs_rmse']:.4f}, "
        f"full={reconstruction['recon_full_rmse']:.4f}"
    )
    print(f"  Evaluation split: {args.eval_split} (model selection split: val)")
    print("=" * 72)

    mask_tag = "argomask" if args.argo_mask else f"mask{round(args.mask * 100):02d}"
    tag = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
    checkpoint = CKPT_DIR / (
        f"lsomf_ca_omf{tag}_h{H_OUT}_{mask_tag}_p{args.prior_mode}_"
        f"mseed{mask_seed}_s{args.seed}.pt"
    )
    checkpoint_metadata = dict(data["metadata"])
    checkpoint_metadata.update(
        {
            # The calendar anchor is the last input month.
            "model_version": MODEL_VERSION,
            "model_seed": args.seed,
            "selection_split": "val",
            "selection_metric": (
                "validation_missing_reconstruction_rmse"
                if args.selection_metric == "reconstruction"
                else "validation_forecast_rmse"
            ),
            "evaluation_split": args.eval_split,
            "validation_used_for_selection": True,
            "early_stopping_disabled": bool(args.disable_early_stop),
            "lr_codec": float(args.lr_codec),
            "lr_ft": float(args.lr_ft),
        }
    )
    torch.save(
        {
            "format": "lsomf-net-checkpoint",
            "model_state": model.state_dict(),
            "metadata": checkpoint_metadata,
        },
        checkpoint,
    )
    result_path = args.result_path or (
        OUT_DIR / (
            f"lsomf_result_{mask_tag}_p{args.prior_mode}_"
            f"mseed{mask_seed}_s{args.seed}.json"
        )
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "metadata": checkpoint_metadata,
        "forecast": results,
        "reconstruction": reconstruction,
    }
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Saved checkpoint: {checkpoint}")
    print(f"  Saved result metadata: {result_path}")


if __name__ == "__main__":
    main()
