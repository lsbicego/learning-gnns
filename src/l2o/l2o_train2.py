"""

export PYTHONPATH=$PYTHONPATH:~/projects/INR  # add INR to PYTHONPATH

Training the baselines, PNA and RT models on FashionMNIST-conv (task 17, see l2o_utils.py).
The hyperparameters (hid, layers, etc) are set based on the best performance on FashionMNIST-conv.

1. Baseline (FF):
    python experiments/mnist/l2o_train.py --train_tasks 17 --gnn scale --hid 64 --layers 2;

2. Baseline (LSTM-FF):
    python experiments/mnist/l2o_train.py --train_tasks 17 --gnn lstm --hid 32 --layers 1;

3. PNA-FF:
    python experiments/mnist/l2o_train.py --train_tasks 17 --gnn pna_noscale --hid 64 --layers 8 --wave_pos_embed;

4. RT-FF:
    python experiments/mnist/l2o_train.py --train_tasks 17 --gnn rt_noscale --hid 32 --layers 2 --wave_pos_embed;

Evaluation on CIFAR-10-conv (task 11, see l2o_utils.py) can be run as:
    python experiments/mnist/l2o_eval.py --ckpt results/l2o_fashionmnist17_.../step_xx.pt --train_tasks 11

- l2o_fashionmnist17_... is the directory of the checkpoint from the training above;
- xx is the step number, chosen based on the performance (test_acc) on FashionMNIST-conv.

"""

import os
import argparse
import numpy as np
import random
import time
from datetime import datetime
import psutil
import subprocess
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models
from torchvision import datasets, transforms
from itertools import chain
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from omegaconf import OmegaConf

from src.l2o.l2o_utils import *

# INR
from src.nn.rt_transformer import RTTransformer, RTransformerParams
from src.nn.gnn import GNNParams
from src.nfn.models import TransferNet


process = psutil.Process(os.getpid())  # for ram calculation


class MetaOpt(nn.Module):
    """
    Predicting two outputs is based on
    https://colab.research.google.com/github/google/learned_optimization/blob/main/docs/notebooks/no_dependency_learned_optimizer.ipynb

    """
    def __init__(self, hid=32,
                 activ=nn.ReLU,
                 rnn=False,
                 momentum=0,
                 preprocess=True,
                 keep_grads=False,
                 lambda12=0.01,
                 gnn='scale',
                 layers=2,
                 heads=2,
                 layer_layout=(784, 20, 10),
                 wave_pos_embed=False,
                 max_kernel_size=5):
        super(MetaOpt, self).__init__()
        self.hid = (hid,) if not isinstance(hid, (tuple, list)) else hid
        self.preprocess = preprocess
        self.keep_grads = keep_grads
        self.lambda12 = lambda12
        self.max_kernel_size = max_kernel_size
        in_features = 2

        if momentum > 1:
            # Based on Section D.1 in https://arxiv.org/pdf/1810.10180.pdf
            momentum = torch.tensor([0.5, 0.9, 0.99, 0.999, 0.9999], requires_grad=False).view(1, -1)
            self.register_buffer('momentum', momentum)
            in_features += 5
        elif momentum > 0:
            # self.momentum = momentum
            # in_features += 1
            raise NotImplementedError('5 momentum values defined above with momentum > 1 is more consistent with '
                                      'the paper (https://arxiv.org/pdf/1810.10180.pdf) and should be better')
        else:
            self.momentum = None

        if preprocess:
            in_features *= 2

        hid_dim = (self.hid[0] // 4) if rnn else self.hid[0]
        if gnn == 'rt_scale':
            print('layer_layout', layer_layout, gnn)
            self.gnn = RTransformerParams(2 if self.momentum is None else (in_features * max_kernel_size ** 2),
                                          hid_dim,
                                          hid_dim,
                                          n_layers=layers,
                                          n_heads=heads,
                                          layer_layout=layer_layout,
                                          out_scale=1.0,
                                          wave_pos_embed=wave_pos_embed)
            hid_dim += in_features
        elif gnn == 'rt_noscale':
            print('layer_layout', layer_layout, gnn)
            self.gnn = RTTransformer(2 if self.momentum is None else (in_features * max_kernel_size ** 2),
                                     hid_dim,
                                     hid_dim,
                                     n_layers=layers,
                                     n_heads=heads,
                                     layer_layout=layer_layout,
                                     wave_pos_embed=wave_pos_embed,
                                     graph_features='weights')
            hid_dim += in_features

        elif gnn == "PNA_LSTM":
            print('layer_layout', layer_layout, gnn)
            d_hid = hid_dim
            d_out = hid_dim
            self.gnn = GNNParams(2 if self.momentum is None else (in_features * max_kernel_size ** 2),
                                 d_hid,
                                 d_out,
                                 "src.nn.gnn.PNA_LSTM",
                                 gnn_kwargs=OmegaConf.create(dict(
                                    in_channels=d_hid,
                                    hidden_channels=d_hid,
                                    num_layers=layers,
                                    out_channels=d_hid,
                                    aggregators=['mean', 'min', 'max', 'std'],
                                    scalers=['identity', 'amplification'],
                                    edge_dim=d_hid,
                                    dropout=0.0,
                                    norm="layernorm",
                                    act="silu",
                                    deg=None,
                                    update_edge_attr=True,
                                    modulate_edges=True,
                                    gating_edges=False,
                                 )),
                                 layer_layout=layer_layout,
                                 rev_edge_features=True,
                                 num_probe_features=0,
                                 zero_out_bias=False,
                                 zero_out_weights=False,
                                 bias_ln=False,
                                 weight_ln=False,
                                 sin_emb=False,
                                 input_layers=1,
                                 use_pos_embed=True,
                                 out_scale=1.0 if gnn == 'pna_scale' else None,
                                 wave_pos_embed=wave_pos_embed,
                                 )
            hid_dim += in_features
        
        
        elif gnn == "GINE":
            print('layer_layout', layer_layout, gnn)
            d_hid = hid_dim
            d_out = hid_dim
            self.gnn = GNNParams(2 if self.momentum is None else (in_features * max_kernel_size ** 2),
                                 d_hid,
                                 d_out,
                                 "src.nn.gnn.GINE",
                                 gnn_kwargs=OmegaConf.create(dict(
                                    in_channels=d_hid,
                                    hidden_channels=d_hid,
                                    num_layers=layers,
                                    out_channels=d_hid,
                                    aggregators=['mean', 'min', 'max', 'std'],
                                    scalers=['identity', 'amplification'],
                                    edge_dim=d_hid,
                                    dropout=0.0,
                                    norm="layernorm",
                                    act="silu",
                                    deg=None,
                                    update_edge_attr=True,
                                    modulate_edges=True,
                                    gating_edges=False,
                                 )),
                                 layer_layout=layer_layout,
                                 rev_edge_features=True,
                                 num_probe_features=0,
                                 zero_out_bias=False,
                                 zero_out_weights=False,
                                 bias_ln=False,
                                 weight_ln=False,
                                 sin_emb=False,
                                 input_layers=1,
                                 use_pos_embed=True,
                                 out_scale=1.0 if gnn == 'pna_scale' else None,
                                 wave_pos_embed=wave_pos_embed,
                                 )
            hid_dim += in_features

        elif gnn == "GAT":
            print('layer_layout', layer_layout, gnn)
            d_hid = hid_dim
            d_out = hid_dim
            self.gnn = GNNParams(2 if self.momentum is None else (in_features * max_kernel_size ** 2),
                                 d_hid,
                                 d_out,
                                 "src.nn.gnn.GAT",
                                 gnn_kwargs=OmegaConf.create(dict(
                                    in_channels=d_hid,
                                    hidden_channels=d_hid,
                                    num_layers=layers,
                                    out_channels=d_hid,
                                    aggregators=['mean', 'min', 'max', 'std'],
                                    scalers=['identity', 'amplification'],
                                    edge_dim=d_hid,
                                    dropout=0.0,
                                    norm="layernorm",
                                    act="silu",
                                    deg=None,
                                    update_edge_attr=True,
                                    modulate_edges=True,
                                    gating_edges=False,
                                 )),
                                 layer_layout=layer_layout,
                                 rev_edge_features=True,
                                 num_probe_features=0,
                                 zero_out_bias=False,
                                 zero_out_weights=False,
                                 bias_ln=False,
                                 weight_ln=False,
                                 sin_emb=False,
                                 input_layers=1,
                                 use_pos_embed=True,
                                 out_scale=1.0 if gnn == 'pna_scale' else None,
                                 wave_pos_embed=wave_pos_embed,
                                 )
            hid_dim += in_features
        elif gnn.startswith('pna'):
            print('layer_layout', layer_layout, gnn)
            d_hid = hid_dim
            d_out = hid_dim
            self.gnn = GNNParams(2 if self.momentum is None else (in_features * max_kernel_size ** 2),
                                 d_hid,
                                 d_out,
                                 "src.nn.gnn.PNA",
                                 gnn_kwargs=OmegaConf.create(dict(
                                    in_channels=d_hid,
                                    hidden_channels=d_hid,
                                    num_layers=layers,
                                    out_channels=d_hid,
                                    aggregators=['mean', 'min', 'max', 'std'],
                                    scalers=['identity', 'amplification'],
                                    edge_dim=d_hid,
                                    dropout=0.0,
                                    norm="layernorm",
                                    act="silu",
                                    deg=None,
                                    update_edge_attr=True,
                                    modulate_edges=True,
                                    gating_edges=False,
                                 )),
                                 layer_layout=layer_layout,
                                 rev_edge_features=True,
                                 num_probe_features=0,
                                 zero_out_bias=False,
                                 zero_out_weights=False,
                                 bias_ln=False,
                                 weight_ln=False,
                                 sin_emb=False,
                                 input_layers=1,
                                 use_pos_embed=True,
                                 out_scale=1.0 if gnn == 'pna_scale' else None,
                                 wave_pos_embed=wave_pos_embed,
                                 )
            hid_dim += in_features

        elif gnn.startswith('gps'):
            print('layer_layout', layer_layout, gnn)
            d_hid = hid_dim
            d_out = hid_dim
            self.gnn = GNNParams(2 if self.momentum is None else (in_features * max_kernel_size ** 2),
                                 d_hid,
                                 d_out,
                                 "src.nn.gnn.GPS",
                                 gnn_kwargs=OmegaConf.create(dict(
                                    in_channels=d_hid,
                                    hidden_channels=d_hid,
                                    num_layers=layers,
                                    out_channels=d_hid,
                                    aggregators=['mean', 'min', 'max', 'std'],
                                    scalers=['identity', 'amplification'],
                                    edge_dim=d_hid,
                                    dropout=0.0,
                                    norm="layernorm",
                                    act="silu",
                                    deg=None,
                                    update_edge_attr=True,
                                    modulate_edges=True,
                                    gating_edges=False,
                                 )),
                                 layer_layout=layer_layout,
                                 rev_edge_features=True,
                                 num_probe_features=0,
                                 zero_out_bias=False,
                                 zero_out_weights=False,
                                 bias_ln=False,
                                 weight_ln=False,
                                 sin_emb=False,
                                 input_layers=1,
                                 use_pos_embed=True,
                                 out_scale=1.0 if gnn == 'pna_scale' else None,
                                 wave_pos_embed=wave_pos_embed,
                                 )
            hid_dim += in_features
        elif gnn.startswith('nfn'):

            # kwargs:
            #   hidden_chan: 128
            #   hidden_layers: 3
            #   mode: NP
            #   out_scale: 0.01
            #   dropout: 0.0
            #   gfft:
            #     in_channels: 1
            #     mapping_size: 128
            #     scale: 3
            #   iosinemb:
            #     max_freq: 3
            #     num_bands: 3
            #     enc_layers: false
            in_dim = 2 if self.momentum is None else (in_features * max_kernel_size ** 2)

            if gnn.find('nogaus_noios') >= 0:
                gfft = None
                iosinemb = None
            else:
                gfft = {'in_channels': in_dim, 'mapping_size': hid_dim, 'scale': 3}
                iosinemb = {'max_freq': 3, 'num_bands': 3, 'enc_layers': False}

            if gnn.lower().find('hnp') >= 0:
                mode = 'HNP'  # TODO: error in "n_in, n_out = network_spec.get_io() AttributeError: 'list' object has no attribute 'get_io'"
            else:
                mode = 'NP'

            print('gfft', gfft, 'iosinemb', iosinemb, 'mode', mode)

            self.gnn = TransferNet(layer_layout=layer_layout,
                                   hidden_chan=hid_dim,
                                   hidden_layers=layers,
                                   mode=mode,
                                   out_scale=None,
                                   out_chan=hid_dim,
                                   in_channels=in_dim,
                                   gfft=gfft,
                                   iosinemb=iosinemb,
                                   )
            hid_dim += in_features
        elif gnn in ['lstm', 'gru']:
            self.gnn_proj = nn.Sequential(nn.Linear(in_features, hid_dim),
                                          activ(),
                                          nn.Linear(hid_dim, hid_dim))
            self.gnn = nn.LSTMCell(hid_dim, hid_dim) if gnn == 'lstm' else nn.GRUCell(hid_dim, hid_dim)
            hid_dim += in_features
        else:
            self.gnn = None

        self.fc = nn.Sequential(
            *chain.from_iterable(
                [
                    [nn.Linear((self.hid[0] if rnn else (hid_dim if self.gnn is not None else in_features))
                               if i == 0 else self.hid[i - 1], h), activ()]
                    for i, h in enumerate(self.hid)
                ]
            ),
            nn.Linear(self.hid[-1], 2),
        )
        self.rnn = None
        if rnn:
            if self.gnn is not None:
                self.in_w = nn.Linear(hid_dim, self.hid[0])
            else:
                self.in_w = nn.Linear(in_features, self.hid[0])
            self.rnn = nn.LSTMCell(self.hid[0], self.hid[0])

    def forward(self, model, hx=None, momentum=None):

        x = []
        for n, p in model.named_parameters():
            if self.keep_grads:
                x.append(torch.stack((p.flatten(), p.grad.data.flatten()), dim=-1))
            else:
                x.append(torch.stack((p.detach().flatten(), p.grad.data.flatten()), dim=-1))
            # if p.detach() is not used, the gradients are preserved for the entire sequence, which creates
            # a discrepancy between training and testing. To avoid such a discrepancy, one can keep the gradients
            # during meta-testing, but that is much more computationally expensive

        x = torch.cat(x, dim=0)  # params and grads

        # Compute momentum features
        if self.momentum is not None:
            # momentum = momentum_value * momentum + grad
            momentum = self.momentum * momentum + x[:, 1:2]
            x = torch.cat((x, momentum), dim=-1)
        if self.preprocess:
            x = preprocess_features(x)

        if self.gnn is not None:
            assert not self.keep_grads, 'not tested'

            if isinstance(self.gnn, (nn.LSTMCell, nn.GRUCell)):
                # Should be similar to "LSTM_FF" from https://arxiv.org/pdf/2009.11243.pdf
                x_gnn_in = self.gnn_proj(x)
                offset = 0
                hx = None
                x_gnn = []
                for n, p in model.named_parameters():
                    k = p.numel()
                    hx = self.gnn(x_gnn_in[offset:offset + k].mean(0, keepdim=True), hx)
                    offset += k
                    x_gnn.append( (hx[0] if isinstance(self.gnn, nn.LSTMCell) else hx).expand(k, -1))
                x_gnn = torch.cat(x_gnn, dim=0)

            else:
                weights, biases, ks = [], [], []
                offset = 0
                for _, p_ in model.named_parameters():
                    k = p_.numel()

                    # Get features for each weight/bias based on x features (momentum/grad/etc) computed above
                    if self.momentum is None:
                        # for backward compatibility with the previous git commit (models trained with momentum=0)
                        p = torch.cat((p_.detach().unsqueeze(-1), p_.grad.detach().unsqueeze(-1)), dim=-1)
                    else:
                        p = x[offset:offset + k].reshape(*p_.shape, -1)  # all features for this layer

                    # Get the features in the format appropriate for the INR code
                    # p:  32,3,3,3,14, where 14 is the feature dimension (momentum/grad/etc)
                    # p_: 32,3,3,3
                    # print('p', p.shape)

                    if len(p_.shape) == 2:
                        # linear layer
                        ks.append((1, 1))
                        p = p.unsqueeze(2).unsqueeze(3)  # unsqueeze here because there are features
                    elif len(p_.shape) == 1:
                        # bias
                        # make features as the first dim, since dims are transposed in transform_weights_biases
                        p = p.permute(1, 0)
                    else:
                        # conv
                        ks.append(tuple(p_.shape[2:]))

                    p = transform_weights_biases(p,
                                                 max_kernel_size=(self.max_kernel_size, self.max_kernel_size),
                                                 linear_as_conv=True)

                    # print('p new', p.shape)
                    # p torch.Size([32, 3, 3, 3, 14])
                    # p new torch.Size([3, 32, 25, 14])
                    # p torch.Size([32, 14])
                    # p new torch.Size([32, 14, 25])
                    # p torch.Size([64, 32, 3, 3, 14])
                    # p new torch.Size([32, 64, 25, 14])
                    # p torch.Size([64, 14])
                    # p new torch.Size([64, 14, 25])
                    # p torch.Size([64, 64, 3, 3, 14])
                    # p new torch.Size([64, 64, 25, 14])
                    # p torch.Size([64, 14])
                    # p new torch.Size([64, 14, 25])
                    # p torch.Size([10, 64, 14])
                    # p new torch.Size([64, 10, 25, 14])
                    # p torch.Size([10, 14])
                    # p new torch.Size([10, 14, 25])

                    if len(p_.shape) == 4:
                        p = p.flatten(2, 3)
                        weights.append(p.unsqueeze(0))
                    elif len(p_.shape) == 2:
                        p = p.flatten(2, 3)
                        weights.append(p.unsqueeze(0))
                    else:
                        # assert len(p_.shape) == 1, p_.shape
                        p = p.flatten(1, 2)
                        biases.append(p.unsqueeze(0))
                    offset += k

                new_w, new_b = self.gnn((weights, biases))
                x_gnn = []
                for w, b, s in zip(new_w, new_b, ks):
                    # We need x_gnn features for each weight including kernels
                    # So as in the lstm baseline, expand/copy the same (1,1) feature to (3,3) or (5,5)
                    # print('w', w.shape, 'b', b.shape, 's', s)
                    # Task 17:
                    # w torch.Size([1, 1, 16, 32]) b torch.Size([1, 16, 32]) s (3, 3)
                    # w torch.Size([1, 16, 32, 32]) b torch.Size([1, 32, 32]) s (3, 3)
                    # w torch.Size([1, 32, 32, 32]) b torch.Size([1, 32, 32]) s (3, 3)
                    # w torch.Size([1, 32, 10, 32]) b torch.Size([1, 10, 32]) s (1, 1)
                    w = w[0].permute(1, 0, 2).unsqueeze(2).unsqueeze(3).expand(
                        -1, -1, *s, -1).reshape(-1, w.shape[-1])
                    # print('w', w.shape)
                    x_gnn.append(w)
                    x_gnn.append(b.squeeze())
                x_gnn = torch.cat(x_gnn, dim=0)

            # print('x_gnn', x_gnn.shape, 'x', x.shape)
            # x_gnn torch.Size([56970, 32]) x torch.Size([56970, 14])
            x = torch.cat((x, x_gnn), dim=-1)

        if self.rnn:
            hx = self.rnn(self.in_w(x), hx)
            outs = self.fc(hx[0] if isinstance(hx, tuple) else hx)
        else:
            outs, hx = self.fc(x), None

        # slice out the last 2 elements
        scale = outs[:, 0]
        mag = outs[:, 1]
        # Compute a step as follows.
        return (scale * self.lambda12 * torch.exp(mag * self.lambda12)), hx, momentum


def eval_meta_opt(meta_opt, test_cfg, seed, args, device, print_interval=20, steps=None, amp=False):
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    net, hx_, momentum_ = init_model(test_cfg, args)
    if isinstance(meta_opt, nn.Module):
        meta_opt.eval()
    else:
        # SGD, Adam, AdamW, etc.
        net.train()
        meta_opt = meta_opt(net.parameters())

    t = time.time()
    max_iters = test_cfg['max_iters'] if steps is None else steps
    train_loader = trainloader_mapping[test_cfg["dataset"]]()
    epochs = int(np.ceil(max_iters / len(train_loader)))
    step = 0
    acc_trace = []  # <<< NEW: list to store test acc over time
    for epoch in range(epochs):
        for _, (x, y) in enumerate(train_loader):
            # net.zero_grad()  # not needed since grads are detached in set_model_params in the next lines
            output = net(x.to(device))
            y = y.to(device)
            loss = F.cross_entropy(output, y)

            loss.backward(retain_graph=args.keep_grads)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc = pred.eq(y.view_as(pred)).sum() / len(y)

            if isinstance(meta_opt, nn.Module):
                with torch.set_grad_enabled(args.keep_grads):
                    with torch.amp.autocast(enabled=amp, device_type=device):  # use amp to reduce memory usage
                        predicted_upd, hx_, momentum_ = meta_opt(net, hx=hx_, momentum=momentum_)
                    set_model_params(net, predicted_upd, keep_grad=args.keep_grads, retain_graph=args.keep_grads)

            else:
                # SGD, Adam, AdamW, etc.
                meta_opt.step()
                meta_opt.zero_grad()

            if (step + 1) % min(100, args.inner_steps) == 0 and step < max_iters - 1:
                test_acc_, test_loss_ = test_model(net, device, testloader_mapping[test_cfg["dataset"]]())
                acc_trace.append(test_acc_)  # <<< NEW: save test acc
                print('test_acc_/test_loss_', test_acc_, test_loss_)

                # reset hidden states and momentum (but not the model/net) to align with the training regime
                # not sure it is needed in the current version, but was important in some preliminary experiments
                # Observation: interestingly, this reset does not change the results significantly
                if (step + 1) % args.inner_steps == 0:
                    net, hx_, momentum_ = init_model(test_cfg, args, net)

            if (step + 1) % print_interval == 0 or step == max_iters - 1:
                r = process.memory_info().rss / 10 ** 9
                g = -1 if device == 'cpu' else (torch.cuda.memory_reserved(0) / 10 ** 9)
                print('Training {} net: seed={}, step={:05d}/{:05d}, train loss={:.3f}, acc={:.3f}, '
                      'speed: {:.2f} s/b, mem ram/gpu: {:.2f}/{:.2f}G'.format(test_cfg['name'],
                                                                              seed,
                                                                              step + 1,
                                                                              max_iters,
                                                                              loss.item(),
                                                                              acc.item(),
                                                                              (time.time() - t) / (step + 1),
                                                                              r,
                                                                              g))
            step += 1
            if step >= max_iters:
                break
        if step >= max_iters:
            break

    test_acc_, test_loss_ = test_model(net, device, testloader_mapping[test_cfg["dataset"]]())
    print("seed: {}, test accuracy: {:.2f}, test loss: {:.4f}\n".format(seed,
                                                                        test_acc_,
                                                                        test_loss_))

    return acc_trace

def eval_meta_opt2(meta_opt, test_cfg, seed, args, device, print_interval=20, steps=None, amp=False):
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    net, hx_, momentum_ = init_model(test_cfg, args)
    if isinstance(meta_opt, nn.Module):
        meta_opt.eval()
    else:
        net.train()
        meta_opt = meta_opt(net.parameters())

    t = time.time()
    max_iters = test_cfg['max_iters'] if steps is None else steps
    train_loader = trainloader_mapping[test_cfg["dataset"]]()
    epochs = int(np.ceil(max_iters / len(train_loader)))
    step = 0
    acc_trace = []  # <<< NEW: list to store test accuracy over time

    for epoch in range(epochs):
        for _, (x, y) in enumerate(train_loader):
            output = net(x.to(device))
            y = y.to(device)
            loss = F.cross_entropy(output, y)

            loss.backward(retain_graph=args.keep_grads)

            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(y.view_as(pred)).sum() / len(y)

            if isinstance(meta_opt, nn.Module):
                with torch.set_grad_enabled(args.keep_grads):
                    with torch.amp.autocast(enabled=amp, device_type=device):
                        predicted_upd, hx_, momentum_ = meta_opt(net, hx=hx_, momentum=momentum_)
                    set_model_params(net, predicted_upd, keep_grad=args.keep_grads, retain_graph=args.keep_grads)
            else:
                meta_opt.step()
                meta_opt.zero_grad()

            step += 1  # increment step after optimizer step

            # <<< NEW: Collect test accuracy every print_interval steps (and at final step)
            if (step) % print_interval == 0 or step == max_iters:
                test_acc_, test_loss_ = test_model(net, device, testloader_mapping[test_cfg["dataset"]]())
                acc_trace.append(test_acc_)
                print(f'test_acc_/test_loss_ {test_acc_}, {test_loss_}')

                # Reset hidden states and momentum if needed
                if step % args.inner_steps == 0:
                    net, hx_, momentum_ = init_model(test_cfg, args, net)

                # Print memory and progress info (already had this)
                r = process.memory_info().rss / 10 ** 9
                g = -1 if device == 'cpu' else (torch.cuda.memory_reserved(0) / 10 ** 9)
                print('Training {} net: seed={}, step={:05d}/{:05d}, train loss={:.3f}, acc={:.3f}, '
                      'speed: {:.2f} s/b, mem ram/gpu: {:.2f}/{:.2f}G'.format(
                    test_cfg['name'], seed, step, max_iters, loss.item(), acc.item(), (time.time() - t) / step, r, g))

            if step >= max_iters:
                break
        if step >= max_iters:
            break

    # Final test accuracy after all steps
    test_acc_, test_loss_ = test_model(net, device, testloader_mapping[test_cfg["dataset"]]())
    print(f"seed: {seed}, test accuracy: {test_acc_:.2f}, test loss: {test_loss_:.4f}\n")

    return acc_trace


def init_config(parser, steps=1000, inner_steps=None, log_interval=1):

    print('starting at', datetime.today())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument('-m', '--model', type=str, default='mlp', choices=['lstm', 'mlp'],
                        help='MetaOpt model')
    parser.add_argument('-H', '--hid', type=int, default=32)
    parser.add_argument('-s', '--steps', type=int, default=steps, help='number of outer steps')
    parser.add_argument('-i', '--inner_steps', type=int,
                        default=parser.parse_known_args()[0].steps if inner_steps is None else inner_steps,
                        help='number of inner/unroll steps')
    parser.add_argument('-t', '--train_tasks', type=str, default='3')  # can potentially train on a combination tasks
    parser.add_argument('--max_kernel_size', type=int, default=5,
                        help='convolutional layer maximum height/width (square kernels are assumed)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=log_interval)
    parser.add_argument('--gnn', type=str, default='scale',
                        # choices=['rt_scale', 'pna_scale', 'rt_noscale', 'pna_noscale', 'scale', 'lstm', 'gru',
                        #          'nfn', 'nfn_nogaus_noios'],
                        help='use gnn from INR')
    parser.add_argument('--layers', type=int, default=2, help='number of layers in the pna/rt')
    parser.add_argument('--heads', type=int, default=4, help='number of heads in the rt')
    parser.add_argument('--wave_pos_embed', action='store_true',
                        help='use wave positional embedding as in Attention is all you need')
    parser.add_argument('-M', '--momentum', type=float, default=5,
                        help='momentum features, 0 means no momentum, >1 means use 5 features from L. Metz')

    parser.add_argument('-l', '--lr', type=float, default=3e-4)
    parser.add_argument('-w', '--wd', type=float, default=1e-4)
    parser.add_argument('--opt', type=str, default='adam')

    # Below are deprecated arguments
    parser.add_argument('--no_preprocess', action='store_true',
                        help='do not preprocess features as in '
                             '"Learning to learn by gradient descent by gradient descent"')
    parser.add_argument('-g', '--keep_grads', action='store_true',
                        help='Keep the gradients w.r.t. the parameters (in addition to the hidden states) '
                             'for the entire sequence. This behavior should be the same for meta-testing as well to '
                             'avoid the train/test mismatch and so perhaps this option is infeasible on large nets.')
    # NOTE: ADDED BY LUCA
    parser.add_argument('--random_inner_steps_after', type=int, default=0, 
                        help='Start randomly sampling inner_steps after this number of outer steps (0 to disable)')
    parser.add_argument('--random_inner_steps_min', type=float, default=0.25,
                        help='Minimum factor for random inner steps (relative to set inner_steps)')
    parser.add_argument('--random_inner_steps_max', type=float, default=3.0, 
                        help='Maximum factor for random inner steps (relative to set inner_steps)')
    
    print('\nEnvironment:')
    env = {}
    try:
        # print git commit to ease code reproducibility
        env['git commit'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(e, flush=True)
        env['git commit'] = 'no git'
    env['hostname'] = platform.node()
    env['torch'] = torch.__version__
    env['torchvision'] = torchvision.__version__
    if env['torch'][0] in ['0', '1'] and not env['torch'].startswith('1.9') and not env['torch'].startswith('1.1'):
        print('WARNING: pytorch >= 1.9 is strongly recommended for this repo!')

    env['cuda available'] = torch.cuda.is_available()
    env['cudnn enabled'] = cudnn.enabled
    env['cuda version'] = torch.version.cuda
    env['start time'] = time.strftime('%Y%m%d-%H%M%S')
    for x, y in env.items():
        print('{:20s}: {}'.format(x[:20], y))

    args = parser.parse_args()
    args.train_tasks = list(map(int, args.train_tasks.split(',')))

    def print_args(args, name):
        print('\n%s:' % name)
        args = vars(args)
        for x in sorted(args.keys()):
            y = args[x]
            print('{:20s}: {}'.format(x[:20], y))
        print('\n', flush=True)

    print_args(args, 'Script Arguments')

    return args, device


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='l2o training')
    # Meta-training arguments

    parser.add_argument('-b', '--meta_batch_size', type=int, default=1,
                        help='number of inner problems (initializations/tasks) over which the gradients are averaged')
    parser.add_argument('--truncate', type=int, default=0)  # truncate gradients of lstm every truncate steps
    parser.add_argument('--progress_steps', type=int, default=0,
                        help='gradually increase the number of inner steps from this value (0 to disable, default)')

    args, device = init_config(parser, steps=1000, inner_steps=100, log_interval=100)

    rnn = args.model.lower() == 'lstm'
    save_dir = 'results/l2o_{}{}_{}{}_{}_lr{:.6f}_wd{:.6f}_mom{:.2f}_hid{}_layers{}_iters{}_innersteps{}{}{}{}'.format(
        TEST_TASKS[args.train_tasks[0]]["dataset"],
        args.train_tasks[0],
        args.model.lower(),
        args.gnn,
        args.opt,
        args.lr,
        args.wd,
        args.momentum,
        args.hid,
        args.layers,
        args.steps,
        args.inner_steps,
        '' if args.no_preprocess else '_preproc',
        '_wave' if args.wave_pos_embed else '',
        '_grads' if args.keep_grads else '')
    print('save_dir: %s\n' % save_dir)

    if os.path.exists(os.path.join(save_dir, "step_999.pt")):
        raise ValueError('Already trained', os.path.join(save_dir, "step_999.pt"))

    seed_everything(args.seed)

    if args.gnn.startswith(('rt', 'pna', 'nfn', "gps")) or args.gnn in ["PNA_LSTM", "GINE", "GAT"]:
        train_cfg_ = TEST_TASKS[np.random.choice(args.train_tasks)]
        model, _, _ = init_model(train_cfg_, args)
        layer_layout = get_layout(model)
        print('assuming fixed model', model, 'layer_layout', layer_layout)
    else:
        assert args.gnn in ['scale', 'lstm', 'gru'], 'unknown gnn'
        layer_layout = None

    metaopt_cfg = dict(hid=[args.hid] * args.layers,
                       rnn=rnn,
                       momentum=args.momentum,
                       preprocess=not args.no_preprocess,
                       keep_grads=args.keep_grads,
                       layer_layout=layer_layout,
                       gnn=args.gnn,
                       layers=args.layers,
                       heads=args.heads,
                       wave_pos_embed=args.wave_pos_embed,
                       max_kernel_size=args.max_kernel_size)
    print('metaopt_cfg', metaopt_cfg)
    metaopt = MetaOpt(**metaopt_cfg).to(device).train()
    print(metaopt, 'params: %d' % sum([p.numel() for p in metaopt.parameters()]))

    if args.opt == 'adam':
        opt_fn = Adam
    elif args.opt == 'adamw':
        opt_fn = AdamW
    else:
        raise NotImplementedError(f'unknown optimizer {args.opt}')

    optimizer = opt_fn(metaopt.parameters(), lr=args.lr, weight_decay=args.wd)
    inner_steps_final = args.inner_steps
    if args.progress_steps > 0:
        # steps = (args.steps // 3, 2 * args.steps // 3, args.steps)  # (10, 20, 30)
        inner_steps_increase = min(args.inner_steps // 3, 200)
        inner_steps = args.progress_steps  # initial inner steps
        steps_rng = np.arange(args.progress_steps, args.inner_steps, inner_steps_increase)
        steps = np.linspace(args.steps // 10, args.steps, len(steps_rng))
        # T_max = int(args.steps / args.inner_steps + args.steps / (args.inner_steps * 2) +
        #             args.steps / (args.inner_steps * 3))
        print('steps', steps, 'steps_rng', steps_rng, 'inner_steps_increase', inner_steps_increase)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    else:
        T_max = args.steps
        inner_steps = args.inner_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

    model = None
    train_cfg = None
    st = time.time()
    train_loaders = {}
    outer_steps_count = 0
    inner_steps_count = 0
    print('\nTraining MetaOpt...')

    while outer_steps_count < args.steps:

        metaopt.train()

        if train_cfg is None:
            seed_everything(outer_steps_count)
            train_cfg = TEST_TASKS[np.random.choice(args.train_tasks)]

        if train_cfg["dataset"] not in train_loaders or train_loaders[train_cfg["dataset"]] is None:
            seed_everything(outer_steps_count)  # to make sure dataloaders are different each time
            train_loaders[train_cfg["dataset"]] = iter(trainloader_mapping[train_cfg["dataset"]]())

        try:
            data, target = next(train_loaders[train_cfg["dataset"]])
        except StopIteration:
            train_loaders[train_cfg["dataset"]] = iter(trainloader_mapping[train_cfg["dataset"]]())
            data, target = next(train_loaders[train_cfg["dataset"]])

        if data is not None:
            data, target = data.to(device), target.to(device)

        if model is None:
            if data is None:
                # quadratic optimization task
                train_cfg["net_args"]['a'], train_cfg["net_args"]['b'] = np.random.rand(2) * 2 + 1

            model, hx, momentum = init_model(train_cfg, args)
            inner_steps_count = 0
        
        # NOTE: ADDED BY US
        # Add randomized inner steps after specified number of steps
        if args.random_inner_steps_after > 0 and outer_steps_count >= args.random_inner_steps_after:
            # Save the original inner_steps value for reference
            if not hasattr(args, 'orig_inner_steps'):
                args.orig_inner_steps = inner_steps
                
            # Randomly sample inner_steps value between min and max factors
            factor = np.random.uniform(args.random_inner_steps_min, args.random_inner_steps_max)
            inner_steps = max(1, int(args.orig_inner_steps * factor))
            print(f"Randomized inner_steps: {inner_steps} (factor: {factor:.2f})")


        if inner_steps > args.truncate > 0 and (outer_steps_count + 1) % args.truncate == 0:
            # reinitialize the hx, which should detach the gradient
            # however, the loss/gradients from the previous steps should still update the metaopt properly
            model, hx, momentum = init_model(train_cfg, args, model)
        # model.zero_grad()  # not needed since grads are detached in p.detach_() in the next lines
        model = model.to(device)
        loss_inner = model().mean() if data is None else F.cross_entropy(model(data), target)
        loss_inner.backward(retain_graph=args.keep_grads)

        w, hx, momentum = metaopt(model, hx=hx, momentum=momentum)  # upd hidden states and get the predicted params
        set_model_params(model, w, keep_grad=True, retain_graph=args.keep_grads)
        # keep_grad is always true to backprop loss_outer through params in the next lines and update metaopt

        # use same data for now, but can be a different batch
        loss_outer = model().mean() if data is None else F.cross_entropy(model(data), target)
        loss_outer = loss_outer / args.meta_batch_size
        loss_outer.backward(retain_graph=True)  # retain graph because we backprop multiple times through metaopt

        if not args.keep_grads:
            # detach to avoid backproping through the params in the next steps
            # backprop through the hidden states (hx) is still happening though
            for p in model.parameters():
                p.detach_()
                p.requires_grad = True  # same as in set_model_params()

        outer_upd = False
        if (inner_steps_count + 1) % inner_steps == 0:

            if (inner_steps_count + 1) % (inner_steps * args.meta_batch_size) == 0:
                # print('upd metaopt', outer_steps_count)
                optimizer.step()  # make a gradient step based on a sequence of inner_steps predictions
                optimizer.zero_grad()
                outer_upd = True
                # can test meta-optimized network for sanity check
                if data is not None:
                    print('test_acc_/test_loss_', '= %.2f / %.3f' % test_model(model, device,
                                                                               testloader_mapping[train_cfg[
                                                                                   "dataset"]]()),
                          'outer step={:03d}/{:03d}'.format(outer_steps_count + 1, args.steps))
                scheduler.step()
            model = None  # to reset the model/initial weights
            train_cfg = None  # to let choose random training tasks

        if args.progress_steps > 0 and (outer_steps_count + 1) == steps[0]:
            inner_steps += inner_steps_increase
            args.inner_steps = inner_steps
            print('progressive training', outer_steps_count + 1,
                  'inner_steps', inner_steps, args.inner_steps,
                  'steps', steps)
            steps = steps[1:]

        if (inner_steps_count + 1) % args.log_interval == 0 or outer_steps_count == args.steps - 1:
            print('Training MetaOpt: '
                  'outer step={:03d}/{:03d}, '
                  'inner step={:03d}/{:03d}, lr={:.3e}, '
                  'loss inner/outer={:.3f}/{:.3f}, '
                  'speed: {:.2f} sec/outer step, mem ram/gpu: {:.2f}/{:.2f}G'.format(outer_steps_count + 1,
                                                                          args.steps,
                                                                          inner_steps_count + 1,
                                                                          inner_steps,
                                                                          scheduler.get_last_lr()[0],
                                                                          loss_inner.item(),
                                                                          loss_outer.item(),
                                                                          (time.time() - st) / (outer_steps_count + 1),
                                                                          process.memory_info().rss / 10 ** 9,
                                                                          -1 if device == 'cpu' else (
                                                                                  torch.cuda.memory_reserved(0) /
                                                                                  10 ** 9)), flush=True)

        if ((outer_steps_count + 1) % 100 == 0 or (outer_steps_count + 1) == args.steps) and outer_upd:
            try:
                checkpoint = {
                    "model_state_dict": metaopt.state_dict(),
                    "step": outer_steps_count,
                    "config": args,
                    "metaopt_cfg": metaopt_cfg
                }

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                file_path = os.path.join(save_dir, f"step_{outer_steps_count}.pt")
                torch.save(checkpoint, file_path)
                print('saving the checkpoint done to %s' % file_path)
            except Exception as e:
                print('error in saving the checkpoint', e)

            if data is not None:
                print('\nEval MetaOpt, task:', TEST_TASKS[args.train_tasks[0]])
                eval_meta_opt(metaopt, TEST_TASKS[args.train_tasks[0]], TEST_SEEDS[0], args, device, print_interval=1)

        inner_steps_count += 1
        if outer_upd:
            outer_steps_count += 1

    print('done!', datetime.today())
