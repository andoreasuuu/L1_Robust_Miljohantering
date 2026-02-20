# K2 – PyTorch Pipelines & Modellträning

## Overview

This project implements a complete Deep Learning pipeline in PyTorch.

The goal was to build a modular and reproducible training setup including:
- Custom Dataset
- DataLoader
- Neural network using nn.Module
- Training and evaluation logic
- Hyperparameter experiments
- Dataset versioning using DVC
- A single reproducible entrypoint (main.py)

---

## Project Structure

dataset.py  
model.py  
train.py  
main.py  
params.yaml  
data/ (tracked with DVC)  
runs.csv  

The code is divided into logical modules as requested.

---

## Dataset & Versioning (DVC)

The project uses CIFAR-10 via torchvision.

The dataset is stored in:

data/cifar10/

The raw dataset files are NOT committed to Git.  
Instead, they are tracked using DVC.

To restore the dataset:

uv run dvc pull

---

## Reproducibility

Install dependencies:

uv sync

Run an experiment:

uv run python main.py --exp exp1

Available experiments:
- exp1
- exp2
- exp3

All hyperparameters are defined in params.yaml.

---

## Model

The model is a simple convolutional neural network with:
- 2 convolution layers
- MaxPooling
- Fully connected classifier
- CrossEntropyLoss
- Adam optimizer

The goal was not to maximize accuracy but to demonstrate a clean and modular ML pipeline.

---

## Experiments (Parameters vs Results)

| Experiment | Epochs | Batch Size | Learning Rate | Weight Decay | Best Validation Accuracy |
|------------|--------|------------|---------------|--------------|--------------------------|
| exp1       | 5      | 64         | 0.001         | 0.0001       | 0.6821                   |
| exp2       | 5      | 128        | 0.001         | 0.0001       | 0.6903                   |
| exp3       | 5      | 64         | 0.0003        | 0.0001       | 0.6322                   |

Results are automatically logged in runs.csv.

---

## Reflection

This project demonstrates how to structure a small ML pipeline in a modular and reproducible way.

Key reflections:

- Separating dataset, model, and training logic improves clarity and maintainability.
- Using params.yaml allows easy hyperparameter experimentation without modifying code.
- Increasing batch size from 64 to 128 improved validation accuracy slightly.
- Lowering the learning rate reduced performance in this setup.
- DVC provides a clean way to handle datasets without polluting Git history.
- A single main.py entrypoint ensures reproducibility.

Compared to implementing everything in a notebook, this structure feels closer to real-world ML workflows.

## (Valfritt) GitHub-repo

Projektet kan även hittas på GitHub:

[](https://github.com/andoreasuuu/L1_Robust_Miljohantering)
