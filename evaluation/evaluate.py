"""Evaluate one or more LSOMF-Net checkpoints on the held-out test set."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_preprocessing.data_protocol import PREPROCESSING_VERSION  # noqa: E402
from data_preprocessing.prepare import ARGO_PREPROCESSING_VERSION, prepare_data  # noqa: E402
from evaluation.metrics import evaluate_final  # noqa: E402
from model.lsomf_net import MODEL_VERSION, get_model  # noqa: E402
from training.train import _set_all_seeds  # noqa: E402

METRICS = ("lead1", "mae_all_lead", "mape_all_lead", "r2_all_lead")
PROTOCOL_KEYS = (
    "data_sha256", "mask_ratio", "mask_seed", "anchor_idx", "n_pca",
    "t_in", "h_out", "train_end", "val_end", "prior_mode",
)
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Processed regional NPZ")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="Trusted checkpoint files")
    parser.add_argument("--argo-mask", type=Path, default=None, help="Argo 3-D observation mask NPZ for Argo checkpoints")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation.json")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    groups = {}
    rows = []
    for checkpoint in args.checkpoints:
        # These files use Python serialization; load only trusted checkpoints.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved = payload["metadata"]
        if saved.get("model_version") != MODEL_VERSION:
            raise ValueError(f"Unsupported checkpoint version: {checkpoint}")
        mode = saved.get("sparse_input_mode", "synthetic_mask")
        if mode not in {"synthetic_mask", "gdcsm_at_raw_argo_mask"}:
            raise ValueError(f"Unsupported sparse input mode: {mode}")
        if mode == "gdcsm_at_raw_argo_mask" and args.argo_mask is None:
            parser.error("--argo-mask is required for Argo-mask checkpoints")
        if mode == "synthetic_mask" and saved.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError(f"Unexpected preprocessing version for synthetic mask: {checkpoint}")
        if mode == "gdcsm_at_raw_argo_mask" and saved.get("preprocessing_version") != ARGO_PREPROCESSING_VERSION:
            raise ValueError(f"Unexpected preprocessing version for Argo mask: {checkpoint}")
        seed = saved.get("model_seed")
        if not isinstance(seed, int):
            raise ValueError(f"Missing integer model seed: {checkpoint}")
        _set_all_seeds(seed)
        data = prepare_data(
            mask_ratio=0.0 if mode == "gdcsm_at_raw_argo_mask" else saved["mask_ratio"], data_path=args.data,
            mask_seed=saved["mask_seed"], anchor_idx=saved["anchor_idx"],
            train_end=saved["train_end"], val_end=saved["val_end"],
            prior_mode=saved["prior_mode"],
            argo_mask_path=args.argo_mask if mode == "gdcsm_at_raw_argo_mask" else None,
        )
        for key in PROTOCOL_KEYS:
            if saved.get(key) != data["metadata"].get(key):
                raise ValueError(f"{checkpoint}: {key} differs from the data or experiment configuration")
        for key in ("pca_solver", "scikit_learn_version"):
            if key in saved and saved[key] != data["metadata"][key]:
                raise ValueError(f"{checkpoint}: {key} differs from the training environment")
        if mode == "gdcsm_at_raw_argo_mask" and "argo_mask_sha256" in saved and saved["argo_mask_sha256"] != data["metadata"]["argo_mask_sha256"]:
            raise ValueError(f"Argo mask differs from checkpoint metadata: {checkpoint}")
        model = get_model().cpu()
        model.load_state_dict(payload["model_state"], strict=True)
        measured = evaluate_final(model, data, eval_split="test")
        if any(not math.isfinite(float(measured[key])) for key in METRICS):
            raise ValueError(f"Non-finite forecast metric: {checkpoint}")
        group_key = tuple(saved[key] for key in PROTOCOL_KEYS) + (mode, data["metadata"].get("argo_mask_sha256"))
        group = groups.setdefault(group_key, [])
        if any(row["model_seed"] == seed for row in group):
            raise ValueError(f"Duplicate model seed in the same experiment: {seed}")
        row = {"checkpoint": str(checkpoint), "model_seed": seed, "forecast": measured}
        group.append(row)
        rows.append(row)
        print(f"mask={saved['mask_ratio']:.4%} seed={seed}: RMSE={measured['lead1']:.8f} m/s", flush=True)

    summary = []
    for key, group in groups.items():
        metrics = {}
        for metric in METRICS:
            values = [float(row["forecast"][metric]) for row in group]
            metrics[metric] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
        entry = {"protocol": dict(zip(PROTOCOL_KEYS + ("sparse_input_mode", "argo_mask_sha256"), key)), "runs": len(group), "metrics": metrics}
        summary.append(entry)
        print(f"mask={entry['protocol']['mask_ratio']:.4%}: {len(group)} runs, mean RMSE={metrics['lead1']['mean']:.8f} m/s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"device": "cpu", "torch": torch.__version__, "summary": summary, "runs": rows}, indent=2), encoding="utf-8")
    print(f"Evaluated {len(rows)} runs; wrote {args.output}")


if __name__ == "__main__":
    main()
