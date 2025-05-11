import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from omegaconf import OmegaConf

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList, Sequential
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor
# from torch_geometric.utils.trim_to_layer import TrimToLayer
from torch_geometric.utils import trim_to_layer

from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.nn.conv import MessagePassing, GPSConv, GINConv, GATConv
from torch_geometric.nn.dense.linear import Linear as pygLinear
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree

from src.nn.rt_transformer import GraphConstructor, create_object
from src.nn.transformer_head import TransformerDecoder


def to_pyg_batch(node_features, edge_features, edge_index):
    # edge_features = edge_features.flatten(1, 2)
    data_list = [
        torch_geometric.data.Data(
            x=node_features[i],
            edge_index=edge_index,
            edge_attr=edge_features[i, edge_index[0], edge_index[1]],
        )
        for i in range(node_features.shape[0])
    ]
    return torch_geometric.data.Batch.from_data_list(data_list)


def nn_to_edge_index(layer_layout, device, dtype=torch.long):
    edge_index = []
    num_nodes = sum(layer_layout)

    node_offset = 0
    nodes_per_layer = []
    for n in layer_layout:
        nodes_per_layer.append(list(range(node_offset, node_offset + n)))
        node_offset += n

    for i in range(1, len(layer_layout)):
        for j in nodes_per_layer[i - 1]:
            for k in nodes_per_layer[i]:
                edge_index.append([j, k])

    return torch.tensor(edge_index, device=device, dtype=dtype).T


def graph_to_wb(edge_features, node_features, weights, biases):
    new_weights = []
    new_biases = []

    start = 0
    for w in weights:
        size = torch.prod(torch.tensor(w.shape[1:-1]))
        new_weights.append(edge_features[:, start:start + size].view(*w.shape[:-1], edge_features.shape[-1]))
        start += size

    start = 0
    for b in biases:
        size = torch.prod(torch.tensor(b.shape[1:-1]))
        new_biases.append(node_features[:, start:start + size].view(*b.shape[:-1], node_features.shape[-1]))
        start += size

    return new_weights, new_biases


class GNN(nn.Module):
    def __init__(
        self,
        d_in,
        d_hid,
        d_out,
        gnn_cls,
        gnn_kwargs,
        layer_layout,
        rev_edge_features,
        graph_features,
        num_probe_features,
        zero_out_bias,
        zero_out_weights,
        bias_ln,
        weight_ln,
        sin_emb,
        input_layers,
        use_pos_embed,
        inp_factor=1,
        compile=False,
        jit=False,
        transformer_kwargs=None,
        normalize=False,
        wave_pos_embed=False,
    ):
        super().__init__()
        # self.rev_edge_features = rev_edge_features
        self.graph_features = graph_features
        self.nodes_per_layer = layer_layout
        self.layer_idx = torch.cumsum(torch.tensor([0] + layer_layout), dim=0)
        # self.layer_layout = [1] * layer_layout[0] + layer_layout[1:]
        # self.layer_layout_for_edge_index = layer_layout
        edge_index = nn_to_edge_index(self.nodes_per_layer, "cpu", dtype=torch.long)
        if rev_edge_features:
            edge_index = torch.cat(
                [edge_index, edge_index.flip(dims=(0,))], dim=-1
            )
        self.register_buffer(
            "edge_index",
            edge_index,
            persistent=False,
        )

        self.construct_graph = GraphConstructor(
            d_in=d_in,
            d_hid=d_hid,
            layer_layout=layer_layout,
            rev_edge_features=rev_edge_features,
            num_probe_features=num_probe_features,
            zero_out_bias=zero_out_bias,
            zero_out_weights=zero_out_weights,
            bias_ln=bias_ln,
            weight_ln=weight_ln,
            sin_emb=sin_emb,
            input_layers=input_layers,
            use_pos_embed=use_pos_embed,
            wave_pos_embed=wave_pos_embed,
            inp_factor=inp_factor,
            normalize=normalize,
        )
        num_graph_features = d_hid
        if graph_features == "cat_last_layer":
            num_graph_features = layer_layout[-1] * d_hid
        elif graph_features == "cat_all_layers":
            num_graph_features = len(layer_layout) * d_hid
        
        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )
        gnn_kwargs = OmegaConf.to_container(gnn_kwargs, resolve=True)
        if gnn_kwargs.get("deg", False) is None:
            deg = dict()
            deg[0] = layer_layout[0]
            for i in range(len(layer_layout) - 1):
                deg[layer_layout[i]] = layer_layout[i + 1]
            deg_tensor = torch.zeros(max(deg.keys()) + 1, dtype=torch.long)
            for k, v in deg.items():
                deg_tensor[k] = v
            gnn_kwargs["deg"] = deg_tensor
        self.gnn = create_object(gnn_cls, **gnn_kwargs)
        if jit:
            self.gnn = torch.jit.script(self.gnn)
        if compile:
            self.gnn = torch_geometric.compile(self.gnn)

        if graph_features == "transformer_decoder":
            self.decoder = TransformerDecoder(**transformer_kwargs)

    def forward(self, inputs):
        node_features, edge_features, _ = self.construct_graph(inputs)

        batch = to_pyg_batch(node_features, edge_features, self.edge_index)
        out_node, out_edge = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
        )
        node_features = out_node.reshape(batch.num_graphs, -1, out_node.shape[-1])

        if self.graph_features == "mean":
            graph_features = node_features.mean(dim=1)
        elif self.graph_features == "max":
            graph_features = node_features.max(dim=1).values
        elif self.graph_features == "last_layer":
            graph_features = node_features[:, -self.nodes_per_layer[-1] :].mean(dim=1)
        elif self.graph_features == "cat_last_layer":
            graph_features = node_features[:, -self.nodes_per_layer[-1] :].flatten(1, 2)
        elif self.graph_features.startswith("layer_"):
            layer = int(self.graph_features.split("_")[1])
            graph_features = node_features[
                :, self.layer_idx[layer] : self.layer_idx[layer + 1]].mean(dim=1)
        elif self.graph_features == "cat_all_layers":
            graph_features = torch.cat([node_features[:, self.layer_idx[i] : self.layer_idx[i + 1]].mean(dim=1)
              for i in range(len(self.nodes_per_layer))], dim=1)
        elif self.graph_features == "transformer_decoder":
            edge_features = out_edge.reshape(batch.num_graphs, -1, edge_features.shape[-1])
            graph_features = torch.cat([node_features, edge_features], dim=1)
            graph_features = self.decoder(graph_features)
            graph_features = graph_features.mean(dim=1)

        return self.proj_out(graph_features)
    


class GNNParams(nn.Module):
    def __init__(
        self,
        d_in,
        d_hid,
        d_out,
        gnn_cls,
        gnn_kwargs,
        layer_layout,
        rev_edge_features,
        num_probe_features,
        zero_out_bias,
        zero_out_weights,
        bias_ln,
        weight_ln,
        sin_emb,
        input_layers,
        use_pos_embed,
        inp_factor=1,
        compile=False,
        jit=False,
        out_scale=0.01,
        wave_pos_embed=False,
    ):
        super().__init__()
        # self.rev_edge_features = rev_edge_features

        self.nodes_per_layer = layer_layout
        self.layer_idx = torch.cumsum(torch.tensor([0] + layer_layout), dim=0)
        # self.layer_layout = [1] * layer_layout[0] + layer_layout[1:]
        # self.layer_layout_for_edge_index = layer_layout
        edge_index = nn_to_edge_index(self.nodes_per_layer, "cpu", dtype=torch.long)
        if rev_edge_features:
            edge_index = torch.cat(
                [edge_index, edge_index.flip(dims=(0,))], dim=-1
            )
        self.register_buffer(
            "edge_index",
            edge_index,
            persistent=False,
        )

        self.construct_graph = GraphConstructor(
            d_in=d_in,
            d_hid=d_hid,
            layer_layout=layer_layout,
            rev_edge_features=rev_edge_features,
            num_probe_features=num_probe_features,
            zero_out_bias=zero_out_bias,
            zero_out_weights=zero_out_weights,
            bias_ln=bias_ln,
            weight_ln=weight_ln,
            sin_emb=sin_emb,
            input_layers=input_layers,
            use_pos_embed=use_pos_embed,
            wave_pos_embed=wave_pos_embed,
            inp_factor=inp_factor,
        )

        self.proj_edge = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )
        self.proj_node = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )
        
        gnn_kwargs = OmegaConf.to_container(gnn_kwargs, resolve=True)
        if gnn_kwargs.get("deg", False) is None:
            deg = dict()
            deg[0] = layer_layout[0]
            for i in range(len(layer_layout) - 1):
                deg[layer_layout[i]] = layer_layout[i + 1]
            deg_tensor = torch.zeros(max(deg.keys()) + 1, dtype=torch.long)
            for k, v in deg.items():
                deg_tensor[k] = v
            gnn_kwargs["deg"] = deg_tensor
        print("..."*20)
        print(gnn_kwargs)
        print("..."*20)
        print(gnn_cls)

        self.gnn = create_object(gnn_cls, **gnn_kwargs)
        # raise ValueError("debugging")
        if jit:
            self.gnn = torch.jit.script(self.gnn)
        if compile:
            self.gnn = torch_geometric.compile(self.gnn)

        # out_scale = 0.01
        self.out_scale = out_scale
        if self.out_scale is not None:
            self.weight_scale = nn.ParameterList([
                nn.Parameter(torch.tensor(out_scale)) for _ in range(len(layer_layout)-1)
            ])
            self.bias_scale = nn.ParameterList([
                nn.Parameter(torch.tensor(out_scale)) for _ in range(len(layer_layout)-1)
            ])

    def forward(self, inputs):
        node_features, edge_features, _ = self.construct_graph(inputs)

        batch = to_pyg_batch(node_features, edge_features, self.edge_index)
        node_out, edge_out = self.gnn(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
        )
        edge_features = edge_out.reshape(edge_features.shape[0], -1, edge_out.shape[-1])
        node_features = node_out.reshape(node_features.shape[0], -1, node_out.shape[-1])
        edge_features = self.proj_edge(edge_features)
        node_features = self.proj_node(node_features)

        weights, biases = graph_to_wb(
                            edge_features=edge_features, 
                            node_features=node_features,
                            weights=inputs[0],
                            biases=inputs[1])

        if self.out_scale is not None:
            weights = [w * s for w, s in zip(weights, self.weight_scale)]
            biases = [b * s for b, s in zip(biases, self.bias_scale)]
        # edge_features = edge_out.reshape(edge_features.shape[:-1], -1)

        # node_features = node_out.reshape(*node_features.shape[:-1], -1)

        return weights, biases
    


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        update_edge_attr: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        self.convs = ModuleList()
        if num_layers > 1:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))
            if isinstance(in_channels, (tuple, list)):
                in_channels = (hidden_channels, hidden_channels)
            else:
                in_channels = hidden_channels
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(in_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            norm_layer = normalization_resolver(
                norm,
                hidden_channels,
                **(norm_kwargs or {}),
            )
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm_layer))
            if jk is not None:
                self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels = num_layers * hidden_channels
            else:
                in_channels = hidden_channels
            self.lin = Linear(in_channels, self.out_channels)

        # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        # self._trim = TrimToLayer()

        # Edge update stuff
        self.update_edge_attr = update_edge_attr
        if update_edge_attr:
            self.edge_update = nn.ModuleList([EdgeMLP(
                edge_dim=kwargs["edge_dim"], node_dim=hidden_channels, 
                act=self.act, norm=norm, norm_kwargs=norm_kwargs) for _ in range(num_layers)])  

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        *,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        xs: List[Tensor] = []
        for i in range(self.num_layers):
            if num_sampled_nodes_per_hop is not None:
                x, edge_index, value = trim_to_layer(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                # x, edge_index, value = self._trim(
                #     i,
                #     num_sampled_nodes_per_hop,
                #     num_sampled_edges_per_hop,
                #     x,
                #     edge_index,
                #     edge_weight if edge_weight is not None else edge_attr,
                # )
                x, edge_index, value = trim_to_layer(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight and self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight,
                                  edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.jk_mode is None:
                break
            if self.act is not None and self.act_first:
                x = self.act(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

            # update edge representations
            if self.update_edge_attr:
                edge_attr = self.edge_update[i](x, edge_index, edge_attr)
                # edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x, edge_attr

    @torch.no_grad()
    def inference(self, loader: NeighborLoader,
                  device: Optional[torch.device] = None,
                  progress_bar: bool = False) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using a
        :class:`~torch_geometric.loader.NeighborLoader`, where
        :class:`~torch_geometric.loader.NeighborLoader` should sample the
        full neighborhood for only one layer.
        This is an efficient way to compute the output embeddings for all
        nodes in the graph.
        Only applicable in case :obj:`jk=None` or `jk='last'`.
        """
        raise NotImplementedError
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training
        # assert not loader.shuffle  # TODO (matthias) does not work :(
        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        x_all = loader.data.x.cpu()
        loader.data.n_id = torch.arange(x_all.size(0))

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                if hasattr(batch, 'adj_t'):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index)[:batch.batch_size]
                if i == self.num_layers - 1 and self.jk_mode is None:
                    xs.append(x.cpu())
                    if progress_bar:
                        pbar.update(1)
                    continue
                if self.act is not None and self.act_first:
                    x = self.act(x)
                if self.norms is not None:
                    x = self.norms[i](x)
                if self.act is not None and not self.act_first:
                    x = self.act(x)
                if i == self.num_layers - 1 and hasattr(self, 'lin'):
                    x = self.lin(x)
                xs.append(x.cpu())
                if progress_bar:
                    pbar.update(1)
            x_all = torch.cat(xs, dim=0)
        if progress_bar:
            pbar.close()
        del loader.data.n_id

        return x_all

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
    

class PNA(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)
    

class GPS(BasicGNN):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        assert in_channels == out_channels, "GPSConv only supports in_channels == out_channels"
        return GPSConv(channels=in_channels,
                       conv=PNAConv(in_channels=in_channels, 
                                    out_channels=out_channels, **kwargs),
                       heads=1,
                       attn_type="performer", 
                       )
    


class PNAConv(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
        \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
        h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
        \right)

    with

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},

    where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
    denote MLPs.

    .. note::

        For an example of using :obj:`PNAConv`, see `examples/pna.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/
        examples/pna.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (List[str]): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers (List[str]): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (torch.Tensor): Histogram of in-degrees of nodes in the training
            set, used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        act (str or callable, optional): Pre- and post-layer activation
            function to use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        train_norm (bool, optional): Whether normalization parameters
            are trainable. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor,
        edge_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        train_norm: bool = False,
        modulate_edges: bool = False,
        gating_edges: bool = False,
        **kwargs,
    ):

        aggr = DegreeScalerAggregation(aggregators, scalers, deg, train_norm)
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.modulate_edges = modulate_edges
        self.gating_edges = gating_edges

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if self.edge_dim is not None:
            if modulate_edges:
                self.edge_encoder = pygLinear(edge_dim, 2 * self.F_in)
            else:
                self.edge_encoder = pygLinear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [pygLinear((3 if edge_dim and not modulate_edges else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [pygLinear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [pygLinear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [pygLinear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = pygLinear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            if self.modulate_edges:
                scale, shift = self.edge_encoder(edge_attr).chunk(2, dim=-1)
                if self.gating_edges:
                    scale = torch.sigmoid(scale)
                    shift = torch.zeros_like(shift)  # NOTE: hacky; fix this
                h = torch.cat([x_i, x_j], dim=-1)
            else:
                edge_attr = self.edge_encoder(edge_attr)
                edge_attr = edge_attr.view(-1, 1, self.F_in)
                edge_attr = edge_attr.repeat(1, self.towers, 1)
                h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        if self.modulate_edges and edge_attr is not None:
            hs = [scale * nn(h[:, i]) + shift for i, nn in enumerate(self.pre_nns)]
        else:
            hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')

    @staticmethod
    def get_degree_histogram(loader: DataLoader) -> Tensor:
        r"""Returns the degree histogram to be used as input for the :obj:`deg`
        argument in :class:`PNAConv`."""
        deg_histogram = torch.zeros(1, dtype=torch.long)
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes,
                       dtype=torch.long)
            d_bincount = torch.bincount(d, minlength=deg_histogram.numel())
            if d_bincount.size(0) > deg_histogram.size(0):
                d_bincount[:deg_histogram.size(0)] += deg_histogram
                deg_histogram = d_bincount
            else:
                assert d_bincount.size(0) == deg_histogram.size(0)
                deg_histogram += d_bincount

        return deg_histogram

    

class EdgeMLP(nn.Module):
    def __init__(self, edge_dim: int, node_dim: int, 
                 act: Callable, norm: Union[str, Callable, None] = None,
                 norm_kwargs: Optional[Dict[str, Any]] = None,):
        super().__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.lin_e = nn.Linear(edge_dim, edge_dim)
        self.lin_s = nn.Linear(node_dim, edge_dim)
        self.lin_t = nn.Linear(node_dim, edge_dim)
        self.act = act
        self.lin1 = nn.Linear(edge_dim, edge_dim)
        # self.norm = normalization_resolver(
        #     norm,
        #     edge_dim,
        #     **(norm_kwargs or {}),
        # )

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        h = self.lin_e(edge_attr) + self.lin_s(x)[edge_index[0]] + self.lin_t(x)[edge_index[1]]
        h = self.act(h)
        h = self.lin1(h)
        return h
        # return self.norm(edge_attr + h)


