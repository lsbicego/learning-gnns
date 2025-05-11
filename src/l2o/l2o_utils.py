import os
import argparse
import numpy as np
import random
from functools import partial
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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # metrics = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.0 * correct / len(test_loader.dataset)
    return test_acc, test_loss


def set_model_params(model, ap, keep_grad=True, retain_graph=False):
    """

    :param model: neural net (nn.Module derivative)
    :param ap: parameter updates
    :param keep_grad: gradients are backproped through params
    :param retain_graph: is used when gradients for all iterations are preserved
    :return:
    """
    offset = 0
    # params = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d, NetQuad)):
            continue
        for name, p in module.named_parameters():
            key = name.split('.')[-1]
            n = p.numel()
            tensor = p + ap[offset: offset + n].view_as(p)
            if keep_grad:
                if retain_graph:
                    tensor.retain_grad()  # to keep the grad of non-leaf tensor
            else:
                tensor = tensor.detach()  # to detach the computational graph from the previous steps
                tensor.requires_grad = True  # necessary to compute p.grad in the next iter
            module.__dict__[key] = tensor  # set the value avoiding the internal logic of PyTorch
            module._parameters[key] = tensor  # to that model.parameters() returns predicted tensors

            offset += n
    return


def preprocess_features(x, p=10):
    # based on https://arxiv.org/pdf/1606.04474.pdf (Appendix A)
    # WARNING: the gradient might be unstable/undefined because of the sign function
    assert x.dim() == 2, x.dim()
    n_feat = x.shape[1]
    x_new = torch.zeros(len(x), n_feat * 2).to(x)
    for dim in range(n_feat):
        mask = torch.abs(x[:, dim]) >= torch.exp(-torch.tensor(p)).to(x)
        ind_small, ind_large = torch.nonzero(~mask).flatten(), torch.nonzero(mask).flatten()
        x_new[ind_small, dim * 2] = torch.zeros(len(ind_small)).to(x) - 1
        x_new[ind_small, dim * 2 + 1] = torch.exp(torch.tensor(p)).to(x) * x[ind_small, dim]
        x_new[ind_large, dim * 2] = torch.log(torch.abs(x[ind_large, dim])) / p
        x_new[ind_large, dim * 2 + 1] = torch.sign(x[ind_large, dim])

    return x_new


def timescale_embedding(iterations):
    """
    Timescale embedding adapted from lmetz work:
    https://github.com/google/learned_optimization/blob/main/learned_optimization/learned_optimizers/mlp_lopt.py#L44

    Args:
        iterations: Timescale vector to be embedded (batch_size, seq_length)

    Returns:
        Timescale embedding with shape (batch_size, seq_length, 11)
    """
    timescales = torch.tensor(
        [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000],
        dtype=torch.float32,
        device=iterations.device,
    )
    x = torch.tile(iterations.unsqueeze(-1), (1, 1, 11)).float()
    embed_x = torch.tanh(x / timescales - 1.0)
    return embed_x


def init_model(cfg, args, model=None, verbose=False):
    if model is None:
        if verbose:
            print('reset model', cfg["net_args"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cfg["net_cls"](**cfg["net_args"]).to(device).train()
    if verbose:
        print('reset model state (%d params)' % sum([p.numel() for p in model.parameters()]))
    hx, cx = None, None
    if args.model.lower() == 'lstm':
        hx = torch.cat([torch.zeros(len(p.flatten()), args.hid).to(p) for p in model.parameters()])
        # can initialize hx, cx as params
        # hx = torch.cat([p.view(-1, 1) for p in model.parameters()])
        # if args.preprocess:
        #     hx = preprocess_features(hx)
        #     hx = hx.tile((1, hid // 2))
        # else:
        #     hx = hx.expand(-1, hid)
        cx = hx.clone()

    mom = torch.cat([torch.zeros(len(p.flatten()), int(max(1, args.momentum))).to(p) for p in model.parameters()]) \
        if args.momentum > 0 else None

    return model, (hx, cx), mom


def pad_and_flatten_kernel(kernel, max_kernel_size):
    full_padding = (
        max_kernel_size[0] - kernel.shape[2],
        max_kernel_size[1] - kernel.shape[3],
    )
    # TODO: padding starts from the last dimension, so kernel.shape[3] should be first,
    #  but for square kernels it should be fine

    padding = (
        *([0] * 2 * (kernel.ndim - 4)),  # do not pad the last (weight feature) dimension if present
        full_padding[0] // 2,
        full_padding[0] - full_padding[0] // 2,
        full_padding[1] // 2,
        full_padding[1] - full_padding[1] // 2,
    )
    padding = np.array(padding)
    padding[padding < 0] = 0  # padding should not be negative (can lead to cut out kernel)
    padding = tuple(padding)
    # print('kernel', kernel.shape, 'padding', padding)
    return F.pad(kernel, padding).flatten(2, 3)


def transform_weights_biases(w, max_kernel_size, linear_as_conv=False):
    """
    Convolutional weights are 4D, and they are stored in the following
    order: [out_channels, in_channels, height, width]
    Linear weights are 2D, and they are stored in the following order:
    [out_features, in_features]

    1. We transpose the in_channels and out_channels dimensions in
    convolutions, and the in_features and out_features dimensions in linear
    layers
    2. We have a maximum HxW value, and pad the convolutional kernel with
    0s if necessary
    3. We flatten the height and width dimensions of the convolutional
    weights
    4. We unsqueeze the last dimension of weights and biases
    TODO: We will also define a maximum number of input and output channels
    for the convolutional layers, and a maximum number of layers
    """
    if w.ndim == 1:
        w = w.unsqueeze(-1)
        return w

    w = w.transpose(0, 1)

    # TODO: Simplify the logic here
    if linear_as_conv:
        if w.ndim == 2:
            w = w.unsqueeze(-1).unsqueeze(-1)
        w = pad_and_flatten_kernel(w, max_kernel_size)
    else:
        w = pad_and_flatten_kernel(w, max_kernel_size) if w.ndim == 4 else w.unsqueeze(-1)

    return w


def get_layout(model, standard=True, conv=False):

    if standard:
        weight_shapes = tuple(w.shape[:2] for w in model.parameters() if len(w.shape) > 1)
        layer_layout = [w[1] for w in weight_shapes] + [weight_shapes[-1][0]]
        # layer_layout = [3, 32, 64, 64, 10] -> 173 nodes
    else:

        # In case of CNNs, these layouts are not consistent with the graphs defined in the INR experiments
        # These worked well for RT, but not PNA
        # These are more like hacks to avoid the original errors in the code when passing conv layers

        weight_shapes = tuple(w.reshape(w.shape[0], -1).shape for w in model.parameters() if len(w.shape) > 1)
        bias_shapes = tuple(b.shape[:1] for b in model.parameters() if len(b.shape) == 1)

        layer_layout_in = [n_out_in[1] for n_out_in in weight_shapes]  # [27, 288, 576, 64]

        if conv and np.any([w.dim() == 4 for w in model.parameters()]):
            # layout may be better for convolutional layers
            layer_layout = layer_layout_in  # [27, 288, 576, 64]

        else:
            # generic layout that works for both mlp and conv models (at least implementation-wise)
            layer_layout = [weight_shapes[0][1]] + [b[0] for b in bias_shapes]  # [27, 32, 64, 64, 10]
            layer_layout[-1] = layer_layout[-1] + max(0, np.sum(layer_layout_in) - np.sum(layer_layout))

    return layer_layout


class NetQuad(nn.Module):
    def __init__(self, a=10, b=1, c=0, c2=0, init=None):
        super(NetQuad, self).__init__()
        if init is None:
            self.params = nn.Parameter((5 * torch.randn(2)).clip(-20, 20))
        else:
            self.params = nn.Parameter(torch.tensor(init).float())  # fix initialization as in AdaHessian paper
        self.fn = lambda params: a * (params[0] - c2) ** 2 + b * (params[1] - c) ** 2

    def forward(self):
        return self.fn(self.params)


class ConvNetTiny(nn.Module):
    def __init__(self, in_channels=3, ks=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=ks, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=ks, stride=1, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.conv(x)


class ConvNetCIFAR(nn.Module):
    def __init__(self, in_channels=3, ks=3, hid=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hid, kernel_size=ks, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hid, out_channels=hid * 2, kernel_size=ks, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=hid * 2, out_channels=hid * 2, kernel_size=ks, stride=1, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hid * 2, 10)
        )

    def forward(self, x):
        return self.conv(x)


class NetMNIST(nn.Module):
    """
    Generic class to construct networks we want to optimize (2 layer MLP with 20 hidden units by default)
    """
    def __init__(self, hid=20, activ=nn.ReLU, conv=False, im_size=14, in_channels=1, num_classes=10):
        super().__init__()
        self.hid = (hid,) if not isinstance(hid, (tuple, list)) else hid
        if conv:
            layer, first_dim, last_dim = (
                nn.Conv2d,
                in_channels,
                self.hid[-1] * int(np.ceil(im_size / (2 ** len(self.hid)))) ** 2,
            )
            layer_args = {"kernel_size": 3, "stride": 2, "padding": 1}
        else:
            layer, first_dim, last_dim = nn.Linear, in_channels * im_size**2, self.hid[-1]
            layer_args = {}

        self.fc = nn.Sequential(
            *chain.from_iterable(
                [
                    [layer(first_dim if i == 0 else self.hid[i - 1], h, **layer_args), activ()]
                    for i, h in enumerate(self.hid)
                ]
                + ([[nn.Flatten()]] if conv else [])
            ),
            nn.Linear(last_dim, num_classes),
        )

    def forward(self, x):
        if isinstance(self.fc[0], nn.Linear):
            x = x.view(len(x), -1)
        return self.fc(x)


def mnist28_trainloader(im_size=28):
    use_cuda = torch.cuda.is_available()
    transform = [] if im_size == 28 else [transforms.Resize((im_size, im_size))]
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                transform + [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        pin_memory=use_cuda,
        num_workers=4,
        batch_size=128,
        shuffle=True,
    )
    return train_loader


def mnist28_testloader(im_size=28):
    use_cuda = torch.cuda.is_available()
    transform = [] if im_size == 28 else [transforms.Resize((im_size, im_size))]
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                transform + [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        pin_memory=use_cuda,
        num_workers=4,
        batch_size=1000,
        shuffle=False,
    )
    return loader


def fashionmnist28_trainloader():
    use_cuda = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        pin_memory=use_cuda,
        num_workers=4,
        batch_size=128,
        shuffle=True,
    )
    return train_loader


def fashionmnist28_testloader():
    use_cuda = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        pin_memory=use_cuda,
        num_workers=4,
        batch_size=1000,
        shuffle=False,
    )
    return loader

def cifar10_trainloader():
    use_cuda = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
                ]
            ),
        ),
        pin_memory=use_cuda,
        num_workers=4,
        batch_size=128,
        shuffle=True,
    )
    return train_loader


def cifar10_testloader():
    use_cuda = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
                ]
            ),
        ),
        pin_memory=use_cuda,
        num_workers=4,
        batch_size=1000,
        shuffle=True,
    )
    return loader


trainloader_mapping = {
    "quad": lambda: [(None, None) for i in range(100)],
    "mnist": mnist28_trainloader,
    "mnist8": partial(mnist28_trainloader, im_size=8),
    "fashionmnist": fashionmnist28_trainloader,
    "cifar10": cifar10_trainloader,
}

testloader_mapping = {
    "quad": lambda: [(None, None) for i in range(100)],
    "mnist": mnist28_testloader,
    "mnist8": partial(mnist28_testloader, im_size=8),
    "fashionmnist": fashionmnist28_testloader,
    "cifar10": cifar10_testloader,
}


TEST_TASKS = [
        # mnist tasks with unseen archs
        {"net_args": {}, "net_cls": NetQuad, "dataset": "quad", "max_iters": 10, "name": "mlp_quad"},
        {"net_args": {"hid": 20, "im_size": 8}, "net_cls": NetMNIST, "dataset": "mnist8", "max_iters": 100, "name": "mlp20_mnist8"},
        {"net_args": {"hid": 20, "im_size": 28}, "net_cls": NetMNIST, "dataset": "mnist", "max_iters": 100, "name": "mlp20_mnist"},
        {"net_args": {"hid": 20, "im_size": 28}, "net_cls": NetMNIST, "dataset": "fashionmnist", "max_iters": 100, "name": "mlp20_fashionmnist"},
        {"net_args": {"hid": (128, 128), "im_size": 28}, "net_cls": NetMNIST, "dataset": "fashionmnist", "max_iters": 100, "name": "mlp128_fashionmnist"},
        {"net_args": {"hid": 20, "activ": torch.nn.Sigmoid, "im_size": 28}, "net_cls": NetMNIST, "dataset": "mnist", "max_iters": 100, "name": "mlpsig_mnist"},
        {"net_args": {"hid": 40, "im_size": 28}, "net_cls": NetMNIST, "dataset": "mnist", "max_iters": 100, "name": "mlp40_mnist"},
        {"net_args": {"hid": (16, 16), "conv": True, "im_size": 28}, "net_cls": NetMNIST, "dataset": "mnist", "max_iters": 100, "name": "conv16x16_mnist"},
        {"net_args": {"hid": (16, 16, 16), "conv": True, "im_size": 28}, "net_cls": NetMNIST, "dataset": "mnist", "max_iters": 100, "name": "conv16x16x16_mnist"},
        {"net_args": {"hid": 32, "conv": True, "im_size": 28}, "net_cls": NetMNIST, "dataset": "mnist", "max_iters": 100, "name": "conv32_mnist"},

        # unseen tasks: cifar10
        # 10
        {
            "net_args": {"hid": (16, 16), "conv": True, "im_size": 32, "in_channels": 3},
            "net_cls": NetMNIST,
            "dataset": "cifar10",
            "max_iters": 400,
            "name": "conv16x16_cifar10"
        },

        # 11
        {
            "net_args": {"in_channels": 3},
            "net_cls": ConvNetCIFAR,
            "dataset": "cifar10",
            "max_iters": 400,
            "name": "convnet_cifar10"
        },

        # 12
        {
            "net_args": {"in_channels": 3, "im_size": 32},
            "net_cls": NetMNIST,
            "dataset": "cifar10",
            "max_iters": 400,
            "name": "mlp20_cifar10"
        },

        # 13
        {
            "net_args": {"hid": (128, 128), "in_channels": 3, "im_size": 32},
            "net_cls": NetMNIST,
            "dataset": "cifar10",
            "max_iters": 400,
            "name": "mlp128_cifar10"
        },

        # 14
        {
            "net_args": {"in_channels": 1, "ks": 3},
            "net_cls": ConvNetTiny,
            "dataset": "fashionmnist",
            "max_iters": 100,
            "name": "convnet_tiny_fashionmnist"
        },

        # 15
        {
            "net_args": {"in_channels": 3, "ks": 3},
            "net_cls": ConvNetTiny,
            "dataset": "cifar10",
            "max_iters": 400,
            "name": "convnet_tiny_cifar10"
        },

        # 16
        {
            "net_args": {"hid": (16, 16), "conv": True, "im_size": 28, "in_channels": 1},
            "net_cls": NetMNIST,
            "dataset": "fashionmnist",
            "max_iters": 100,
            "name": "conv16x16_fashionmnist"
        },

        # 17
        {
            "net_args": {"in_channels": 1, "hid": 16},
            "net_cls": ConvNetCIFAR,
            "dataset": "fashionmnist",
            "max_iters": 100,
            "name": "convnet_fashionmnist"
        },
    ]

TEST_SEEDS = [101, 102, 103, 104, 105]
