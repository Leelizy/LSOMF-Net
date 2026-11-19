# LSOMF-Net

**Lightweight Spatiotemporal Prediction of Underwater Sound Speed Fields from Nonuniform Sparse Observations**

This repository contains the PyTorch implementation of **LSOMF-Net**, a lightweight framework for predicting three-dimensional underwater sound speed fields (SSFs) from nonuniform sparse historical observations. The model combines **Observation Context Gated Correction (OCGC)** for historical-state refinement and **Cycle-Aware Orthogonal Moment Fusion (CA-OMF)** for next-month prediction.

> The datasets and pretrained model weights are not included in this repository.

## Repository structure

```text
LSOMF-Net/
|-- data_preprocessing/    # Data preprocessing and sparse observation generation
|-- model/                 # LSOMF-Net architecture
|-- training/              # Model training
|-- evaluation/            # Model evaluation
|-- config.py              # Experimental configuration
|-- .gitignore
`-- README.md
```

## Getting started

### Environment

The manuscript reports experiments using Python 3.10.8 and PyTorch 2.1.2 with CUDA 12.1 on an NVIDIA GeForce RTX 4080 SUPER GPU. The main Python dependencies are PyTorch, NumPy, and scikit-learn.

An environment can be created using:

```bash
conda create -n lsomf python=3.10
conda activate lsomf
pip install numpy==1.26.4 scikit-learn==1.6.0
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

A fresh installation with these versions has not been verified. The original scikit-learn version has not been independently established; this implementation uses PCA with `svd_solver="covariance_eigh"`.

### Data preparation

The experiments use the South Pacific and Azores subregions of the GDCSM_Argo dataset. Each regional NPZ file must contain an `svel` array of shape `[264, 58, 16, 16]` (time x depth x latitude x longitude), covering consecutive months from January 2004 to December 2025. If a `time` array is supplied, the code checks its monthly sequence; otherwise, it assumes this starting month.

For the commands below, place the regional files under `data/`:

```text
data/
|-- south_pacific_data.npz
`-- azores_data.npz
```

The source of the exact regional NPZ files used in the experiments has not been independently established. The `data/` directory is excluded by `.gitignore`.

### Training and evaluation

Train LSOMF-Net on the South Pacific dataset with a 50% missing-data rate:

```bash
python training/train.py --data-path data/south_pacific_data.npz --mask 0.5 --seed 11
```

The artificial missing-data rate can be set to `0.0`, `0.5`, or `0.9`. Training uses two-stage optimization with validation-based early stopping and maximum lengths of 40 and 50 epochs. Checkpoints are selected using validation-set forecast RMSE and saved under `outputs/training/best_checkpoints/`.

Evaluate the trained model on the held-out test set:

```bash
python evaluation/evaluate.py --data data/south_pacific_data.npz --checkpoints outputs/training/best_checkpoints/lsomf_ca_omf_h1_mask50_pfull_mseed1000004_s11.pt
```

The evaluation script reports RMSE, MAE, MAPE, and R^2. Multiple checkpoints can be provided to obtain the mean and standard deviation across runs. Results are saved to `outputs/evaluation.json`.

To run experiments on the Azores subregion, replace the dataset path accordingly. The default mask seed is `1000004`; different model seeds use the same artificial mask at a given missing rate. The manuscript reports 20-run means, so results from a different number of runs are not directly comparable.

### Argo sampling-mask experiment

The Argo experiment requires an additional NPZ file containing a binary `observation_mask` array with the same `[264, 58, 16, 16]` ordering as the regional data. The mask is not included. It specifies Argo observation locations; the sound-speed values at those locations come from the GDCSM field.

```bash
python training/train.py --data-path data/azores_data.npz --argo-mask data/azores_argo_mask.npz --seed 11 --checkpoint-tag azores
python evaluation/evaluate.py --data data/azores_data.npz --argo-mask data/azores_argo_mask.npz --checkpoints outputs/training/best_checkpoints/lsomf_ca_omf_azores_h1_argomask_pfull_mseed1000004_s11.pt
```
