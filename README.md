# Differentially Private Latent Diffusion Models

This repository contains an implementation of the methods described in [Differentially Private Latent Diffusion Models](https://openreview.net/pdf?id=FLOxzCa6DS). The code is based off a public implementation of Latent Diffusion Models, available [here](https://github.com/CompVis/latent-diffusion) (commit `a506df5`).

# Setting Up Your Enviroment:
This project uses Conda as its package management tool which can downloaded [here](https://docs.conda.io/en/latest/). Once installed, clone the repository. The remainder of this document will assume the project is stored in a directory called `DP-LDM`.

**Important**: We strongly recommend using the [Mamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) solver for Conda as it dramatically speeds up environment creation.

```sh
cd DP-LDM/
conda env create -f environment.yaml
conda activate ldm
```

# Pretrained Model Checkpoints:

**Under Construction**

We provide checkpoints of our models with hyperparameters presented in the paper. These checkpoints can be used as-is, or used for pretraining/finetuning your own LDMs. Instructions for how to train your own models is presented below.

## Pretrained Autoencoders (non-private)

| Model | Link |
|-------|------|
|       |      |

## Pretrained LDMs (non-private)

| Model | Link |
|-------|------|
|       |      |

## Fine-tuned LDMs (private)

| Model | Link |
|-------|------|
|       |      |

# Training Your Own Models

Once you have chosen a public/private dataset pair, there are three steps to training your own differentially private latent diffusion models. In each step, you will need to create a configuration file that specifies the hyperparameter of each model. Example config files can be found in `DP-LDM/configs/`.

**Step 1: Autoencoder Pretraining**
```
CUDA_VISIBLE_DEVICES=0 python main.py --base <path to autoencoder yaml> -t --gpus 0,
```

**Step 2: LDM Pretraining**
```
CUDA_VISIBLE_DEVICES=0 python main.py --base <path to dm yaml> -t --gpus 0,
```

**Step 3: Private Fine-tuning**

**Important:** Due to implementation constraints, this step can only be run on a single GPU, specified by the `--accelerator gpu` command line argument.
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --base <path to fine-tune yaml> \
    -t \
    --gpus 0, \
    --accelerator gpu
```

# Sampling

To sample from class-conditional models (e.g. MNIST, FMNIST, CIFAR10):
```bash
python sampling/cond_sampling_test.py \
    -y path/to/config.yaml \
    -ckpt path/to/checkpoint.ckpt \
    -c 0 1 2 3 4 5 6 7 8 9
```

To sample from unconditional models (e.g. CelebA):
```bash
python sampling/unonditional_sampling.py \
    --yaml path/to/config.yaml \
    --ckpt path/to/checkpoint.ckpt
```

# Evaluation

We evaulated our models using two metrics. Code for both is available in the repository. For both methods, first follow the section above to generate sufficiently many samples from your model.

## Downstream Classification Accuracy
For MNIST, to compute the accuracy, the command is :
```bash
python scripts/dpdm_downstreaming_classifier_mnist.py \
    --train path/to/generated_train_images.pt \
    --test path/to/real_test_images.pt
```

We also provide a script that combines sampling and accuracy computation
```bash
python scripts/mnist_sampling_and_acc.py \
    --yaml path/to/config.yaml \
    --ckpt path/to/checkpoint.ckpt
```


```bash
python txt2img.py \
    --yaml path/to/config.yaml \
    --ckpt path/to/checkpoint.ckpt \
    --n_samples 30000 \
    --outname txt2img_samples.pt
```

## FID

First, compute Inception network statistics for the real dataset
```bash
python fid/compute_dataset_stats.py \
    --dataset ldm.data.celeba.CelebATrain \
    --args size:32 \
    --output celeba_train_stats.npz
```

Next, compute the statistics for the generated samples:
```bash
python fid/compute_samples_stats.py \
    --samples celeba32_samples.pt \
    --output celeba_samples_stats.npz
```

Finally, compute FID:
```bash
python fid/compute_fid.py \
    --path1 celeba32_train_stats.npz \
    --path2 celeba32_samples_stats.npz
```

# Implementation Comments

We build our code on top of the [Latent Diffusion](https://github.com/CompVis/latent-diffusion) repository. Thanks to the authors for open sourcing their code! We also borrow techniques from [Transferring Pretrained Diffusion Probabilistic Models](https://openreview.net/forum?id=8u9eXwu5GAb), and would like to thank the authors for privately sending us their code before making it public.

## Differences from `latent-diffusion`

* Moved the implementation of the `DDPM` class to a new file `ddpm_base.py`
* Moved callbacks from `main.py` to `callbacks/*.py`
* Added `glob.escape` to log folder parsing to support special characters
* Changed name of checkpoint created on exception from `last.ckpt` to `on_exception.ckpt`
* Changed name of checkpoint created on signal from `last.ckpt` to `on_signal.ckpt`
