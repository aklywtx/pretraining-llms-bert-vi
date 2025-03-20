# BERT Model Implementation

## Overview

This repository provides a PyTorch implementation of the BERT model for language modeling tasks, specifically Masked Language Modeling (MLM) and Sentence Order Prediction (SOP)/Next Sentence Prediction (NSP).


## Repository Structure

```
.
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
├── trainer
│   ├── __init__.py
│   └── pretrain.py
└── evaluation
    ├── __init__.py
    ├── entailment.py
    ├── faithfulness.py
    ├── real_world_knowledge.py
    ├── winograd.py
    └── evaluation_data
        ├── Winograd.csv
        ├── entailment.csv
        ├── faithfulness.csv
        └── repron_vn_binary.csv
```

## Dataset

The Vietnamese SOP and NSP datasets for training the model are created using the `dataset_creation.py` script.


## Requirements

- Python 3.9+
- torch
- torch_optimizer
- transformers
- datasets
- tqdm
- wandb

## Evaluation Pipelines

### Winograd Schema Challenge

- Processes the input sequence and generates output probabilities for each token position.
- At the masked token position, the model applies a softmax layer to produce a probability distribution over the entire vocabulary.
- Compares the probabilities of the correct and incorrect tokens.
- The token with the higher probability is selected as the model’s predicted choice.

### Faithfulness detection and Textual entailment

- Extracts the [CLS] token embeddings for the input, incorrect sentence, and correct sentence.
- Computes the cosine similarity between the input and both the incorrect and correct sentences.
- Selects the sentence with the higher similarity score.

### Real-world knowledge

- Follows a similar approach to the **Winograd Schema Challenge**, but the correct and incorrect choices are spans of tokens rather than single tokens.
- Computes the probability of each span as the mean probability of the tokens within it.
- The span with the higher mean probability is chosen as the model’s predicted output.


## Usage

### Training

To train the BERT model:

```bash
python -m pretraining-llms-bert-vi --language "vi" --dataset "nsp" --max_length 256 --n_layers 8 --num_heads 8 --embed_dim 512 --lr 1e-4 --batch_size 128 --ff_dropout 0 --id 0 --special "adam_wd_lrd" --output_dir ""
```

Adjust parameters (`epochs`, `batch_size`, `lr`, etc.) as needed.

### Evaluation
Winograd Schema Challenge
```bash
python evaluation/winograd.py
```
Faithfulness detection
```bash
python evaluation/faithfulness.py
```
Real-world knowledge
```bash
python evaluation/entailment.py
```
Textual entailment
```bash
python evaluation/real_world_knowledge.py
```
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
