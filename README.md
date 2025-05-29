# opt_project


First, create an enviornment like so:

```bash
conda create -n opt-project python=3.10
conda activate opt-project
pip install -r requirements.txt
pip install -e . # to install src package for easier imports
```

Please keep requirements up to date.


To run the experiment, you can do:

# Run Commands

Train different L2O models. These are the best settings found by the original authors. 

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

For Evlaluation, run this script with the desired tasks.


```bash
# Evaluation on CIFAR-10
python -m src.l2o.l2o_eval --ckpt results/l2o_fashionmnist17_.../step_xxx.pt --train_tasks 11

```

I added various new settings, two of which could be interesting: 
```bash
# Train a simpler and more popular GINE Graph Conv model
python -m src.l2o.l2o_train --train_tasks 17 --gnn GINE --hid 64 --layers 8

# PNA-LSTM: Train an PNA GNN that uses a LSTM cell for each node --> models temporal AND structural dyanmics, seems to converge well

# NOTE: need to check best settings since this our own model
python -m src.l2o.l2o_train --train_tasks 17 --gnn PNA_LSTM --hid 64 --layers 4