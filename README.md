# ADAGRAD-FUSION

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-orange)](https://pytorch.org/)

This repository contains a PyTorch implementation of the paper "L-BiTMix: A Lightweight Channel-Mix Temporal Convolutional Network for ECG Analysis". It provides tools, training pipelines and utilities to 
## Table of Contents
- [Key features](#-key-features)
- [Requirements](#️-requirements)
- [Quickstart](#-quickstart)
- [Project structure](#-project-structure)
- [Important files and folders](#️-important-files-and-folders)
- [Reproducing experiments](#-reproducing-experiments)
- [Citation](#-citation)

## Key features

## Requirements

This project targets Python 3.8+ and PyTorch 1.10+. Install required packages with:

```powershell
pip install -r requirements.txt
```

Refer to `requirements.txt` for exact package versions used in experiments.

## Quickstart

1. Prepare or download downstream datasets (examples used in the paper): Chapman-Shaoxing, PTB-XL, G12EC.

Quick download the four downsteam datasets: 
```powershell
  wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
```

2. The model was pre-trained on the CODE-15 dataset, which can be downloaded from:
(https://zenodo.org/records/4916206)

3. The pre-trained backbones are available on [Hugging Face], which can be downloaded from:
(https://huggingface.co/KAZABANA/Foundation-Models-for-ECG-classification/tree/main).

4. Preprocess raw data into the project's format with:

```powershell
python datacollection.py
```


## Project structure

```
AdaGrad-Fusion/
├── datacollection.py        # data preprocessing and dataset splitting
├── main.py                 # main experiment entrypoint
├── pipeline_ecg.py         # fine-tuning pipeline
├── evaluation.py           # evaluation and metrics
├── helper_code.py          # utility helpers
├── model_src_ecg/          # model definitions
├── pretrained_checkpoint/  # example checkpoints (not all included)
└── result/                 # experiment outputs (.npy, logs)
```

## Important files and folders

- `datacollection.py` - converts raw ECG datasets into HDF5 and splits them for experiments
- `main.py` - run experiments and adjust hyperparameters
- `pipeline_ecg.py` - contains training logic and evaluation hooks
- `model_src_ecg/` - model definitions
  
## Reproducing experiments

1. Install dependencies.
2. Prepare datasets and convert them with `datacollection.py`.
3. Run `main.py` or the tailored pipeline scripts to reproduce fine-tuning and evaluation.

## Citation

Please cite the L-BiTMix paper when using this code in academic work. 
