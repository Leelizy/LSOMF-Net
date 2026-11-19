"""Configuration for the LSOMF-Net experiment."""
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

CACHE_DIR = PROJECT_ROOT / "data"
CACHE_FILE = "south_pacific_data.npz"
OUT_DIR = PROJECT_ROOT / "outputs" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GH, GW = 16, 16
N_DEP = 58

# Index 21 is the 220 dbar anchor; changing it requires retraining.
ANCHOR_IDX = 21
N_B, N_R, N_ST = 1, 8, 9

T_IN = 12
H_OUT = 1

# Split boundaries refer to target timestamps.
TRAIN_END = 192
VAL_END = 216

# Keep the mask seed independent of the model seed.
MASK_SEED = 1_000_004

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4
LR_CODEC = 5e-4
LR_FT = 1e-4
LAMBDA_REC = 0.1
LAMBDA_RAW_REC = 1.0
LAMBDA_OBS = 1.0
CODEC_EPOCHS = 40
FT_EPOCHS = 50
