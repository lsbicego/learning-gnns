# TODO:
remove train2 and eval2 (i.e., switch names)

# Learning to Optimize with GNNs 

In our project, we extended the implementation and added various experiments around "Neural Graphs" [Kofinas et al. (2024)](https://arxiv.org/abs/2403.12143) for Learning to Optimize. 

A MetaOptimizer is a model that learns how to optimize another neural network --- usually termed the "optimizee" --- on another task. Usually MetaOptimizers are given the gradients of the optimizee and output a treated version that lead to improved convergence over handcrafted optimizers. Neural Graphs (CITE) transform the optimizee into a graph with node and edge features. GNN-based optimizer take the resulting graph structure into account when proposing parameter updates, which yields better performances versus more traditional learning to optimize (L2O) appraoches, that usually consider each parameter independently. 

We first replicated existing experiments and then extended the experiments in two ways. (1) we propose a RecurrentGNN formulation for the L2O task and we train a model randomizing the number of inner training step. (2) We investigate the generalizability of learned optimizers, hypothesising that GNN-based methods do better compared to parameter-based methods when the testing loss landscape differs from the training loss landscape. 

## Setup

First, create an enviornment like so:

```bash
conda create -n opt-project python=3.10
conda activate opt-project
pip install -r requirements.txt
pip install -e . # to install src package for easier imports
```

## Experiments

#### Replication

These commands train the two GNN-based methods (pna_noscale and rt_noscae) and the two L2O baselines (scale, lstm) with the best parameters reported in [Kofinas et al. (2024)](https://arxiv.org/abs/2403.12143) on the FashionMNIST dataset.

```bash
# Baseline (FF)
python -m src.l2o.l2o_train --train_tasks 17 --gnn scale --hid 64 --layers 2

# LSTM-FF Baseline
python -m src.l2o.l2o_train --train_tasks 17 --gnn lstm --hid 32 --layers 1

# PNA-FF (Best architecture from paper)
python -m src.l2o.l2o_train --train_tasks 17 --gnn pna_noscale --hid 64 --layers 8 --wave_pos_embed

# RT-FF
python -m src.l2o.l2o_train --train_tasks 17 --gnn rt_noscale --hid 32 --layers 2 --wave_pos_embed

```

#### Suggested Improvements

```bash
# Recurrent GNN
python -m src.l2o.l2o_train --train_tasks 17 --gnn PNA_LSTM --hid 64 --layers 4

# GNN-variable
python -m src.l2o.l2o_train2 --train_tasks 17 --gnn pna_noscale --hid 64 --layers 8 --wave_pos_embed --random_inner_steps_after 100 --random_inner_steps_min 0.75 --random_inner_steps_max 2.5

```



#### Evaluation & Generalizability

In the original paper, models have been evaluated on a single test task, optimizing a similarly-sized optimizee as seen during training on the CIFAR10 dataset. To run this eval:

```bash
# Evaluation on CIFAR-10
python -m src.l2o.l2o_eval --ckpt results/l2o_fashionmnist17_.../step_xxx.pt --train_tasks 11
```


Changing the argument `train_task` allows the more fine-grained evaluation procedure: 

Changing the hidden dimension:
- **Task 19**: CNN with 8 filters per layer (4× narrower than standard)
- **Task 20**: CNN with 16 filters per layer (2× narrower than standard)
- **Task 11**: CNN with 32 filters per layer (reference architecture)
- **Task 21**: CNN with 64 filters per layer (2× wider than standard)
- **Task 22**: CNN with 128 filters per layer (4× wider than standard)

Changing the network depth:
- **Task 23**: CNN with 1 convolutional layer (minimal depth)
- **Task 24**: CNN with 2 convolutional layers (reduced depth)
- **Task 11**: CNN with 3 convolutional layers (reference architecture)
- **Task 25**: CNN with 4 convolutional layers (increased depth)
- **Task 26**: CNN with 5 convolutional layers (maximum depth)

## Code Structure
```bash
src/
├── l2o/
│   ├── l2o_eval.py        # Evaluation of trained meta-optimizers
│   ├── l2o_train.py       # Main training script for meta-optimizers
│   └── l2o_utils.py       # Utility functions for training/evaluation
├── nfn/
│   ├── models.py          # Neural Field Network implementations
│   ├── common/            # Common utilities for NFN
│   └── layers/            # Custom layer implementations
└── nn/
    ├── gnn.py             # Graph Neural Network implementations
    ├── rt_transformer.py  # Routing Transformer implementation
    └── transformer_head.py # Transformer attention mechanisms
```

- l2o/: Learning-to-Optimize implementation with training and evaluation scripts
- nfn/: Neural Graph models and utilities
- nn/: Core neural network implementations including GNN models

Training is done using l2o_train.py, while evaluation is performed with l2o_eval.py.

## Attribution
The code is adapted & cleaned from the repository shared on [Openreview](https://openreview.net/forum?id=oO6FsMyDBt). The official repository shared in the main paper does not yet include L2O experiments!
