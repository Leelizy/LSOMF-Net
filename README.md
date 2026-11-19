# LSOMF-Net

**Lightweight Spatiotemporal Prediction of Underwater Sound Speed Fields from Nonuniform Sparse Observations**

This repository contains the PyTorch implementation of **LSOMF-Net**, a lightweight framework for predicting three-dimensional underwater sound speed fields (SSFs) from nonuniform sparse historical observations. The model combines **Observation Context Gated Correction (OCGC)** for historical-state refinement and **Cycle-Aware Orthogonal Moment Fusion (CA-OMF)** for next-month prediction.

> The datasets and pretrained model weights are not included in this repository.

---

## Repository structure
```text
LSOMF-Net/
├── data_preprocessing/    # Data preprocessing and sparse observation generation
├── model/                 # LSOMF-Net architecture
├── training/              # Model training
├── evaluation/            # Model evaluation
├── config.py              # Experimental configuration
├── .gitignore
└── README.md
```

---

## Getting started

### Environment

The experiments were conducted using Python 3.10.8 and PyTorch 2.1.2 with CUDA 12.1 on an NVIDIA GeForce RTX 4080 SUPER GPU.

The main Python dependencies are:
```
torch  numpy  scikit-learn
```

A dedicated environment can be created using:
```bash
conda create -n lsomf python=3.10
conda activate lsomf
pip install numpy==1.26.4 scikit-learn==1.6.0
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### Data preparation

The experiments use the South Pacific and Azores subregions of the GDCSM_Argo dataset. Each regional NPZ file should contain an `svel` array of shape `[264, 58, 16, 16]` (time × depth × latitude × longitude).

Place the regional data files under `data/`:
```
data/
├── south_pacific_data.npz
└── azores_data.npz
```

### Training and evaluation

Train LSOMF-Net on the South Pacific dataset with a 50% missing-data rate:
```bash
python training/train.py --data-path data/south_pacific_data.npz --mask 0.5 --seed 11
```

The artificial missing-data rate can be set to `0.0`, `0.5`, or `0.9`. Training uses two-stage optimization with validation-based early stopping.

Trained checkpoints are saved under `outputs/training/best_checkpoints/`.

Evaluate the trained model on the held-out test set:
```bash
python evaluation/evaluate.py --data data/south_pacific_data.npz --checkpoints outputs/training/best_checkpoints/lsomf_ca_omf_h1_mask50_pfull_mseed1000004_s11.pt
```

The evaluation script reports RMSE, MAE, MAPE, and R². Multiple checkpoints can be provided to obtain the mean and standard deviation across independent runs.

Evaluation results are saved to `outputs/evaluation.json`.

To run experiments on the Azores subregion, replace the dataset path accordingly.
