# BERT Model Implementation

## Overview

This repository provides a PyTorch implementation of the BERT model for language modeling tasks, specifically Masked Language Modeling (MLM) and Sentence Order Prediction (SOP)/Next Sentence Prediction (NSP).


## Structure

```
├── __main__.py
├── attention
│   ├── __init__.py
│   └── multihead.py
├── dataset
│   ├── __init__.py
│   └── dataset_mlmsop.py
├── embedding
│   ├── __init__.py
│   ├── bert.py
│   ├── position.py
│   ├── segment.py
│   └── token.py
├── model
│   ├── __init__.py
│   ├── bert.py
│   ├── bertlm.py
│   └── transformer.py
└── trainer
    ├── __init__.py
    └── pretrain.py
```

## Requirements
- Python 3.9+
- torch
- torch_optimizer
- transformers
- datasets
- tqdm
- wandb
- datasets

## Usage

### Training

To train the BERT model:

```bash
python -m vi --language "vi" --dataset "nsp" --max_length 256 --n_layers 8 --num_heads 8 --embed_dim 512 --lr 1e-4 --batch_size 128 --ff_dropout 0 --id 0 --special "adam_wd_lrd" --output_dir ""
```

Adjust parameters (`epochs`, `batch_size`, `lr`, etc.) as needed.

## Logging and Monitoring

Training progress and metrics are logged using [Weights & Biases (wandb)](https://wandb.ai/site). Ensure you have wandb set up properly:

```
pip install wandb
wandb login
```

## Acknowledgements
This codebase references and adapts from:
- [transformer-mlm](https://github.com/rishub-tamirisa/transformer-mlm)
- [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
