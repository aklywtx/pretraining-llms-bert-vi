### Requirements
You have a [wandb][https://wandb.ai/site/] account to monitor the training process. You need to change your wandb API key in the submit file `vi.sub`.
You have a huggingface API token to upload the model to your huggingface account. 

### Create Docker Image
1. create the docker file. You can use the `Dockerfile`.
2. Download Docker(https://docs.docker.com/)
3. Register your project in LST docker registry, explained here: https://wiki2.coli.uni-saarland.de/doku.php?id=user:cluster:docker_registry. You need to use VPN for this.
4. Run the following commands in your local terminal with Docker running:
   ```bash
   docker build --platform=linux/amd64 -f Dockerfile --build-arg USER_UID=$UID --build-arg USER_NAME=$(id -un) -t docker.coli.uni-saarland.de/YOUR_USERNAME/IMAGE_NAME:TAG .
   ```
   For example, I use:
   ```bash
   docker build --platform=linux/amd64 -f Dockerfile --build-arg USER_UID=$UID --build-arg USER_NAME=$(id -un) -t docker.coli.uni-saarland.de/xtong/vibert:v0 .
   ```
5. Push the image to the registry:
   ```bash
   docker push docker.coli.uni-saarland.de/xtong/YOUR_USERNAME/IMAGE_NAME:TAG
   ```
   Now if you login your project in LST docker registry, you should see your docker image.

### Connect to cluster
1. Login 
```bash
# after login
ssh submit # ssh to submit node
```
2. Choose a folder to save model checkpoints. You can use scratch if you have one.
```bash
cd scratch
mkdir YOUR_USERNAME
```
Otherwise set a folder for saving checkpoints, model output & error messages.
Let's say you have a output folder named `/scratch/YOUR_USERNAME`, make some directories in your folder.
```bash
mkdir -p BERT/vi/output
mkdir -p BERT/pl/output
mkdir -p BERT/ka/output
mkdir cache
```

### Train the model on cluster
example: vietnamese
1. Change hyperparameters in `train_vi.sh`
```bash
vim train_vi.sh
# press i to insert
# press esc and :wq to save the file and quit
```
2. Modify the submit file `vi.sub`
```bash
vim vi.sub
# press i to insert
# press esc and :wq to save the file and quit
```
Submit file looks like:
Change the folder names. You could user your wandb key.
```bash
universe                = docker
docker_image            = docker.coli.uni-saarland.de/YOUR USERNAME/IMAGE_NAME:TAG .
initialdir              = /nethome/YOUR_USERNAME
executable              = /nethome/YOUR_USERNAME/BERT/train_vi.sh
output                  = /YOUR OUTPUT_FOLDER/BERT/vi/output/debug.out
error                   = /YOUR_OUTPUT_FOLDER/BERT/vi/output/debug.err
log                     = /YOUR_OUTPUT_FOLDER/BERT/vi/output/debug.log
request_GPUs = 1
request_CPUs = 1
request_memory = 100G
requirements = (TARGET.Machine == "hopper-1.coli.uni-saarland.de" || \
                TARGET.Machine == "hopper-2.coli.uni-saarland.de" || \
                TARGET.Machine == "hopper-3.coli.uni-saarland.de")
environment = TRANSFORMERS_CACHE=/scratch/YOUR_USERNAME/cache; HF_HOME=/scratch/xtong/cache; WANDB_API_KEY=56a6decf76189630d0b38fcc2371477be1d9de43
queue 1
```
3. Submit the job
https://wiki2.coli.uni-saarland.de/doku.php?id=user:cluster:condor
```bash
condor_submit vi.sub
condor_q # check your job status
condor_nodestate # check available GPUs
```


### How to upload the model to huggingface
```bash
./upload_model_to_hf.sh $huggingface_model_name $local_checkpoint_path $huggingface_id`
```


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
python -m vi --language "vi" --dataset "nsp" --max_length 256 --n_layers 8 --num_heads 8 --embed_dim 512 --lr 1e-4 --batch_size 128 --ff_dropout 0 --id 0 --special "adam_wd_lrd" --output_dir "/scratch/xtong"
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
- [Transformer MLM by rishub-tamirisa](https://github.com/rishub-tamirisa/transformer-mlm)
- [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
