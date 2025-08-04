# Temporal Dynamics Enhancer for Spiking Neural Networks (SNNs)

This repository implements the **Temporal Dynamics Enhancer (TDE)** to improve object detection performance in Spiking Neural Networks (SNNs).  
TDE enhances temporal modeling capabilities through the **Spiking Encoder (SE)** and **Attention Gating Module (AGM)**, while reducing energy consumption via the **Spike-Driven Attention (SDA)** mechanism.

## Environment Setup

Please install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Dataset Preparation
Download the official preprocessed datasets:

- VOC  
- EvDET200K

After downloading, update the dataset paths in the configuration files located in the `./config` directory.


## Training
python train.py

## Evaluation
python test.py