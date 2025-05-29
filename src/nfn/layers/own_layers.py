    

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.nn.inits import reset

from src.nn.rt_transformer import RTAttention
import torch.nn as nn

from typing import Union, Tuple, Optional


import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch

class GINEConvEdge(MessagePassing):
    r"""
    GINEConv augmented with an edge‐update MLP.

    For nodes:
        x'_i = node_nn( (1+ε)·x_i + ∑_{j∈N(i)} ReLU(x_j + e_{j,i}) )

    For edges:
        e'_{j,i} = edge_nn(e_{j,i})

    Args:
        node_nn (torch.nn.Module): maps node features [-1, F_in] → [-1, F_out]
        edge_nn (torch.nn.Module): maps edge features [-1, D] → [-1, D_out]
        eps (float, optional): initial ε value  (default: 0.)
        train_eps (bool, optional): if True, ε is learnable  (default: False)
        edge_dim (int, optional): if set, linearly project edge_attr → F_in
        **kwargs: additional MessagePassing args (e.g. aggr)
    """
    def __init__(
        self,
        node_nn: torch.nn.Module,
        edge_nn: torch.nn.Module,
        eps: float = 0.,
        train_eps: bool = False,
        edge_dim: int = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.node_nn = node_nn
        self.edge_nn = edge_nn

        # eps parameter
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))

        # optional linear to project edge_attr → node feature dim
        if edge_dim is not None:
            # infer in_channels from first layer of node_nn
            base_nn = node_nn[0] if isinstance(node_nn, torch.nn.Sequential) else node_nn
            in_channels = getattr(base_nn, 'in_features', getattr(base_nn, 'in_channels', None))
            if in_channels is None:
                raise ValueError("Can't infer in_channels for edge projection")
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.node_nn)
        reset(self.edge_nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # propagate node messages
        if isinstance(x, Tensor):
            x = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        # final node update
        out_x = self.node_nn(out)

        # edge update MLP
        if edge_attr is not None:
            # optional projection to match dims
            if self.lin is not None:
                edge_attr = self.lin(edge_attr)
            out_edge = self.edge_nn(edge_attr)
        else:
            out_edge = None

        return out_x, out_edge

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # project edge_attr if needed
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError(
                "Node and edge feature dims do not match; set `edge_dim`."
            )
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(node_nn={self.node_nn}, '
            f'edge_nn={self.edge_nn})'
        )


import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import (
softmax,
)
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

class GATConvEdge(MessagePassing):
    r"""GATConv extended with an edge‐update MLP.

    Args:
        in_channels (int or tuple): as in GATConv
        out_channels (int): as in GATConv
        edge_nn (torch.nn.Module): MLP to update edge features [E, D_in]→[E, D_out]
        heads (int): number of attention heads
        concat (bool): whether to concat or average heads
        negative_slope (float): LeakyReLU slope
        dropout (float): attention dropout
        add_self_loops (bool)
        edge_dim (int, optional): if not None, proj edge_attr→heads*out_channels
        fill_value (float|Tensor|str)
        bias (bool)
        residual (bool)
        **kwargs: extra MessagePassing args
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_nn: torch.nn.Module,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: int = None,
        fill_value: float = 'mean',
        bias: bool = True,
        residual: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual

        # Node transformations (same as original GATConv)
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, bias=False,)
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin = None

        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        # Existing edge‐attention projection:
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels,
                                   bias=False, weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        # Residual & bias
        total_out = heads * out_channels if concat else out_channels
        if residual:
            dst_dim = in_channels if isinstance(in_channels, int) else in_channels[1]
            self.res_lin = Linear(dst_dim, total_out, bias=False,
                                  weight_initializer='glorot')
        else:
            self.register_parameter('res_lin', None)
        if bias:
            self.bias = Parameter(torch.zeros(total_out))
        else:
            self.register_parameter('bias', None)

        # --- NEW: edge‐update MLP ---
        self.edge_nn = edge_nn

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'lin') and self.lin is not None:
            self.lin.reset_parameters()
        if hasattr(self, 'lin_src'):
            self.lin_src.reset_parameters()
        if hasattr(self, 'lin_dst'):
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res_lin is not None:
            self.res_lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        if self.att_edge is not None:
            torch.nn.init.xavier_uniform_(self.att_edge)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        # reset your edge_nn:
        if hasattr(self.edge_nn, 'reset_parameters'):
            self.edge_nn.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: bool = False,
    ):
        # 1.) Run original GATConv forward to get node embeddings (& attn):
        if return_attention_weights:
            out_x, (ei, alpha) = super().forward(
                x, edge_index, edge_attr, size, return_attention_weights=True)
        else:
            out_x = super().forward(x, edge_index, edge_attr, size,
                                    return_attention_weights=False)

        # 2.) Run edge‐update MLP on **original** edge_attr:
        if edge_attr is not None:
            out_edge = self.edge_nn(edge_attr)
        else:
            out_edge = None

        # 3.) Return both:
        if return_attention_weights:
            return (out_x, (ei, alpha)), out_edge
        else:
            return out_x, out_edge

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, dim_size=None):
        # reuse existing logic for attention:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if edge_attr is not None and self.lin_edge is not None:
            e = edge_attr.view(-1, 1) if edge_attr.dim() == 1 else edge_attr
            e = self.lin_edge(e).view(-1, self.heads, self.out_channels)
            alpha = alpha + (e * self.att_edge).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.in_channels}, {self.out_channels}, heads={self.heads})')



class OwnGPSConv(torch.nn.Module):
    r"""The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper.

    The GPS layer is based on a 3-part recipe:

    1. Inclusion of positional (PE) and structural encodings (SE) to the input
       features (done in a pre-processing step via
       :class:`torch_geometric.transforms`).
    2. A local message passing layer (MPNN) that operates on the input graph.
    3. A global attention layer that operates on the entire graph.

    .. note::

        For an example of using :class:`GPSConv`, see
        `examples/graph_gps.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        graph_gps.py>`_.

    Args:
        channels (int): Size of each input sample.
        conv (MessagePassing, optional): The local message passing layer.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        dropout (float, optional): Dropout probability of intermediate
            embeddings. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        attn_type (str): Global attention type, :obj:`multihead` or
            :obj:`performer`. (default: :obj:`multihead`)
        attn_kwargs (Dict[str, Any], optional): Arguments passed to the
            attention layer. (default: :obj:`None`)
    """
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        elif attn_type == 'performer':
            self.attn = PerformerAttention(
                channels=channels,
                heads=heads,
                **attn_kwargs,
            )
        elif attn_type == "RT":
            self.attn = RTAttention(
                d_node=channels,
                d_edge=channels,
                d_hid=channels,
                n_heads=heads,
                **attn_kwargs,
            )
        else:
            # TODO: Support BigBird
            raise ValueError(f'{attn_type} is not supported')

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()


    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, edge_attr=edge_attr, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask,
                             need_weights=False)
        elif isinstance(self.attn, PerformerAttention):
            h = self.attn(h, mask=mask)
        elif isinstance(self.attn, RTAttention):
            h = self.attn(h, edge_attr, mask=mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads}, '
                f'attn_type={self.attn_type})')
