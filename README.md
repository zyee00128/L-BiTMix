# ADAGRAD-FUSION

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-orange)](https://pytorch.org/)

This repository provides a PyTorch implementation of the paper "L-BiTMix: A Lightweight Channel-Mix Temporal Convolutional Network for ECG Analysis". It offers tools for data preprocessing, model training, fine-tuning, and evaluation for ECG signal analysis, with an organized project structure for easy customization and extension.

## Table of Contents
- [Key features](#-key-features)
- [Requirements](#️-requirements)
- [Quickstart](#-quickstart)
- [Project structure](#-project-structure)
- [Important files and folders](#️-important-files-and-folders)
- [Reproducing experiments](#-reproducing-experiments)
- [Cross-validation details](#-cross-validation-details-key)
- [Citation](#-citation)

## Key features
- Implements the lightweight L-BiTMix temporal convolutional model
- Supports preprocessing and formatting of major ECG datasets (Chapman-Shaoxing, PTB-XL, G12EC)
- Fine-tuning and evaluation pipelines for downstream tasks
- Integrated pre-trained models for easy reproduction and transfer learning
- Highly modular for fast modification and extension

## Requirements

This project targets Python 3.8+ and PyTorch 1.10+. Install dependencies with:
```powershell
pip install -r requirements.txt
```
See `requirements.txt` for the exact package versions required for experiments.

## Quickstart

1. Prepare or download downstream datasets (such as Chapman-Shaoxing, PTB-XL, G12EC used in the paper).
2. For example, to quickly download the PhysioNet Challenge 2021 dataset:
    ```powershell
    wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
    ```
3. The model was pre-trained on the CODE-15 dataset, available from [Zenodo](https://zenodo.org/records/4916206).
4. Pre-trained model weights are available from [Hugging Face](https://huggingface.co/KAZABANA/Foundation-Models-for-ECG-classification/tree/main).
5. Preprocess raw data into the project format with:
    ```powershell
    python datacollection.py
    ```

## Project structure

```
AdaGrad-Fusion/
├── datacollection.py        # data preprocessing and dataset splitting
├── main.py                  # main experiment entrypoint
├── pipeline_ecg.py          # fine-tuning pipeline
├── evaluation.py            # evaluation and metrics
├── helper_code.py           # utility helpers
├── model_src_ecg/           # model definitions
├── pretrained_checkpoint/   # example checkpoints (not all included)
└── result/                  # experiment outputs (.npy, logs)
```

## Important files and folders

- `datacollection.py` - Converts raw ECG datasets into HDF5 and splits them for experiments
- `main.py`           - Run experiments and adjust hyperparameters
- `pipeline_ecg.py`   - Contains training logic and evaluation hooks
- `model_src_ecg/`    - Model definitions
  
## Reproducing experiments

1. Install dependencies.
2. Prepare datasets and convert them with `datacollection.py`.
3. Run `main.py` or the tailored pipeline scripts to reproduce fine-tuning and evaluation.

## Cross-validation details

In `datacollection.py`, within the function `ECGdataset_prepare_finetuning_sepe(args)`:

- The project uses sklearn's `KFold` for 5-fold cross-validation:
    ```python
    KFold(n_splits=5, shuffle=True, random_state=args.seed)
    ```
- For each fold:
    - The train subset (from `train_index`) is further split into a new train and validation subset according to `args.finetune_label_ratio`.
    - The test subset for that fold is defined by the corresponding `test_index`.

Conclusion: The default configuration uses **5-fold cross-validation**.  
To switch to 10-fold cross-validation, simply change `n_splits=5` to `n_splits=10`.  
No other code changes are required; the pipeline will automatically adapt.

## Citation

Please cite the L-BiTMix paper if you use this code or model in your research.
