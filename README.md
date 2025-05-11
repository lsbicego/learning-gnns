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

```bash
# Train on FashionMNIST (task 17) with a RT-FF architecture
python -m src.l2o.l2o_train --train_tasks 17 --gnn rt_noscale --hid 32 --layers 2 --wave_pos_embed

# Or for a PNA-FF architecture
python -m src.l2o.l2o_train --train_tasks 17 --gnn pna_noscale --hid 64 --layers 8 --wave_pos_embed

# Evaluating on CIFAR-10 (task 11)
python -m src.l2o.l2o_eval --ckpt results/l2o_fashionmnist17_.../step_xxx.pt --train_tasks 11
```
