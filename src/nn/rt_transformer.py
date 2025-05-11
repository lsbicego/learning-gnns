import math
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# from xformers.components import attention
# from xformers.components import MultiHeadDispatch

# from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# import torch_geometric
# from omegaconf import OmegaConf
# from nn import inr
# from nn.graph_construct import GaussianFourierFeatureTransform


def create_object(name, **kwargs):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**kwargs)


def batch_to_graphs(weights, biases, weights_mean=None, weights_std=None, 
                    biases_mean=None, biases_std=None):
    device = weights[0].device
    bsz = weights[0].shape[0]
    # for w in weights:
    #     print('w', w.shape)
    # for b in biases:
    #     print('b', b.shape)
    num_nodes = sum(w.shape[1] for w in weights) + weights[-1].shape[2]
    # print('num_nodes', num_nodes)  # must be equal to the sum of terms in layer_layout

    node_features = torch.zeros(bsz, num_nodes, biases[0].shape[-1], device=device)
    edge_features = torch.zeros(bsz, num_nodes, num_nodes, weights[0].shape[-1], device=device)

    row_offset = 0
    col_offset = weights[0].shape[1]  # no edge to input nodes
    for i, w in enumerate(weights):
        _, num_in, num_out, _ = w.shape  # 27,32 -> 288,64
        w_mean = weights_mean[i] if weights_mean is not None else 0
        w_std = weights_std[i] if weights_std is not None else 1
        edge_features[
            :, row_offset : row_offset + num_in, col_offset : col_offset + num_out
        ] = (w - w_mean) / w_std
        row_offset += num_in
        col_offset += num_out

    row_offset = weights[0].shape[1]  # no bias in input nodes
    for i, b in enumerate(biases):
        # biases correspond to rows (out channels)
        _, num_out, _ = b.shape
        b_mean = biases_mean[i] if biases_mean is not None else 0
        b_std = biases_std[i] if biases_std is not None else 1
        node_features[:, row_offset : row_offset + num_out] = (b - b_mean) / b_std
        row_offset += num_out

    return node_features, edge_features


def graphs_to_batch(edge_features, node_features, weights, biases):
    new_weights = []
    new_biases = []

    row_offset = 0
    col_offset = weights[0].shape[1]  # no edge to input nodes
    for w in weights:
        _, num_in, num_out, _ = w.shape
        new_weights.append(edge_features[
            :, row_offset : row_offset + num_in, col_offset : col_offset + num_out
        ])
        row_offset += num_in
        col_offset += num_out

    row_offset = weights[0].shape[1]  # no bias in input nodes
    for b in biases:
        _, num_out, _ = b.shape
        new_biases.append(node_features[:, row_offset : row_offset + num_out])
        row_offset += num_out

    return new_weights, new_biases


def inputs_to_params(weights, biases):
    
    params = []
    for w, b in zip(weights, biases):
        params.append(w)
        params.append(b)
    return params


def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class SinusoidalPosEmb(nn.Module):  # from transformer, PE[2i] = sin((pos/10000)**(2i/dim))
    def __init__(self, dim, freq=1e4, fact=1e6):
        super().__init__()
        self.dim = dim
        self.freq = freq
        self.fact = fact

    def forward(self, x):  # (bs,)
        half_dim = self.dim // 2
        emb = math.log(self.freq) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = self.fact * x * emb.view(*[1] * (x.ndim - 1), half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb  # (bs, dim)

class GraphProbeFeatures(nn.Module):
    def __init__(self, d_in, num_inputs, inr_d_out, input_init=None, proj_dim=None):
        super().__init__()
        # TODO fix hard coded
        n_layers = 3
        up_scale = 16 if d_in == 2 else 32
        inr_module = inr.INR_AF(in_dim=d_in, n_layers=n_layers, up_scale=up_scale, out_channels=inr_d_out)
        fmodel, params = inr.make_functional(inr_module)

        vparams, vshapes = inr.params_to_tensor(params)
        self.sirens = torch.vmap(inr.wrap_func(fmodel, vshapes))
        
        inputs = input_init if input_init is not None else 2 * torch.rand(1, num_inputs, d_in) - 1
        self.inputs = nn.Parameter(inputs, requires_grad=input_init is None)

        # NOTE hard coded maps
        self.reshape_w0 = Rearrange("b i h0 1 -> b (h0 i)")
        self.reshape_w1 = Rearrange("b h0 h1 1 -> b (h1 h0)")
        self.reshape_w2 = Rearrange("b h1 h2 1 -> b (h2 h1)")

        self.reshape_b0 = Rearrange("b h0 1 -> b h0")
        self.reshape_b1 = Rearrange("b h1 1 -> b h1")
        self.reshape_b2 = Rearrange("b h2 1 -> b h2")

        self.proj_dim = proj_dim
        if proj_dim is not None:
            self.proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_inputs, proj_dim),
                    nn.LayerNorm(proj_dim),
                    # nn.SiLU(),
                    # nn.Linear(proj_dim, proj_dim)
                    ) for _ in range(n_layers+1)])


    def forward(self, weights, biases):
        params_flat = torch.cat(
            [self.reshape_w0(weights[0]),
            self.reshape_b0(biases[0]),
            self.reshape_w1(weights[1]),
            self.reshape_b1(biases[1]),
            self.reshape_w2(weights[2]),
            self.reshape_b2(biases[2])], dim=-1)

        out = self.sirens(params_flat, self.inputs.expand(params_flat.shape[0], -1, -1))
        if self.proj_dim is not None:
            out = [proj(out[i].permute(0, 2, 1)) for i, proj in enumerate(self.proj)]
            out = torch.cat(out, dim=1)
            return out
        else:
            out = torch.cat(out, dim=-1)
            return out.permute(0, 2, 1)
        

class GraphConstructor(nn.Module):
    def __init__(self,
                d_in,
                d_hid,
                layer_layout,
                rev_edge_features=False,
                num_probe_features=0,
                zero_out_bias=False,
                zero_out_weights=False,
                bias_ln=False,
                weight_ln=False,
                inp_factor=1,
                input_layers=1,
                sin_emb=False,
                use_pos_embed=True,
                wave_pos_embed=False,
                normalize=None):
        super().__init__()
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
        self.zero_out_bias = zero_out_bias
        self.zero_out_weights = zero_out_weights
        self.use_pos_embed = use_pos_embed
        self.normalize = normalize

        self.pos_embed_layout = (
            [1] * layer_layout[0] + layer_layout[1:-1] + [1] * layer_layout[-1]
        )
        if wave_pos_embed:
            # To facilitate generalization to larger models at test time
            max_len = 10000  # some very large number of positions to make sure it works for most of the cases
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_hid, 2) * (-math.log(10000.0) / d_hid))
            pe = torch.zeros(max_len, d_hid)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_embed = nn.Parameter(pe)
        else:
            self.pos_embed = nn.Parameter(torch.randn(len(self.pos_embed_layout), d_hid))

        # print('wave_pos_embed', wave_pos_embed, self.pos_embed.shape, self.pos_embed.min(), self.pos_embed.max())
        proj_weight = []
        proj_bias = []
        if sin_emb:
            proj_weight.append(GaussianFourierFeatureTransform(d_in, 128, inp_factor))
            # proj_weight.append(SinusoidalPosEmb(4*d_hid, fact=inp_factor))
            proj_weight.append(nn.Linear(256, d_hid))
            proj_bias.append(GaussianFourierFeatureTransform(d_in, 128, inp_factor))
            # proj_bias.append(SinusoidalPosEmb(4*d_hid, fact=inp_factor))
            proj_bias.append(nn.Linear(256, d_hid))
        else:
            # if d_in = 1, then there are 3 features
            # if d_in = 2, then there are 6 features
            proj_weight.append(nn.Linear(int(d_in*(3 if rev_edge_features else 1)), d_hid))
            proj_bias.append(nn.Linear(d_in, d_hid))
            # proj_weight.append(nn.LayerNorm(d_hid))
            # proj_bias.append(nn.LayerNorm(d_hid))

        for i in range(input_layers-1):
            proj_weight.append(nn.SiLU())
            proj_weight.append(nn.Linear(d_hid, d_hid))
            proj_bias.append(nn.SiLU())
            proj_bias.append(nn.Linear(d_hid, d_hid))

        self.proj_weight = nn.Sequential(*proj_weight)
        self.proj_bias = nn.Sequential(*proj_bias)

        self.proj_node_in = nn.Linear(d_hid, d_hid)
        self.proj_edge_in = nn.Linear(d_hid, d_hid)
        print('num_probe_features', num_probe_features)
        if num_probe_features > 0:
            self.gpf = GraphProbeFeatures(
                d_in=layer_layout[0], num_inputs=num_probe_features, 
                inr_d_out=layer_layout[-1], proj_dim=d_hid)
            # self.proj_gpf = nn.Sequential(
            #     nn.LayerNorm(num_probe_features),
            #     nn.Linear(num_probe_features, d_hid),
            # )
        else:
            self.gpf = None

        # if bias_ln:
        #     self.bias_ln = nn.LayerNorm((sum(layer_layout), 1), elementwise_affine=False)
        # else:
        #     self.bias_ln = None
        # if weight_ln:
        #     self.weight_ln = nn.LayerNorm((sum(layer_layout), sum(layer_layout), 3 if rev_edge_features else 1), elementwise_affine=False)
        # else:
        #     self.weight_ln = None
        # DWSNet version
        if normalize == "dwsnet_mnist":
            self.weights_mean = [-0.0001166215879493393, -3.2710825053072767e-06, 7.234242366394028e-05]
            self.weights_std = [0.06279338896274567, 0.01827024295926094, 0.11813738197088242]
            self.biases_mean = [4.912401891488116e-06, -3.210141949239187e-05, -0.012279038317501545]
            self.biases_std = [0.021347912028431892, 0.0109943225979805, 0.09998151659965515]
        # NFN version
        elif normalize == "nfn_mnist":
            self.weights_mean = [-0.00010918354382738471, 4.154461521466146e-07, -5.62107416044455e-05]
            self.weights_std = [0.28673890233039856, 0.01709533855319023, 0.06388015300035477]
            self.biases_mean = [0.0004948264104314148, -0.00015013274969533086, -0.03146892413496971]
            self.biases_std = [0.4084731340408325, 0.1042051613330841, 0.09648718684911728]
        # NFN cifar10
        elif normalize == "nfn_cifar10":
            self.weights_mean = [-0.0003794827207457274, 5.305922059051227e-06, 0.00013588865112978965]
            self.weights_std = [0.2799842357635498, 0.01765584573149681, 0.054600153118371964]
            self.biases_mean = [-0.0003265937266405672, -0.00019049091497436166, -0.0015163210919126868]
            self.biases_std = [0.4089966416358948, 0.10387048870325089, 0.08673509210348129]
        else:
            self.weights_mean = torch.zeros(100)
            self.weights_std = torch.ones(100)
            self.biases_mean = torch.zeros(100)
            self.biases_std = torch.ones(100)

    def forward(self, inputs):
        node_features, edge_features = batch_to_graphs(*inputs,
                                                       weights_mean=self.weights_mean, 
                                                       weights_std=self.weights_std, 
                                                       biases_mean=self.biases_mean, 
                                                       biases_std=self.biases_std)
        # mask currently unused
        mask = edge_features.sum(dim=-1, keepdim=True) != 0
        # mask = mask & mask.transpose(-1, -2)
        if self.rev_edge_features:
            # NOTE doesn't work together with other features anymore
            rev_edge_features = edge_features.transpose(-2, -3)
            # print('rev_edge_features', rev_edge_features.shape, edge_features.shape)
            edge_features = torch.cat(
                [edge_features, rev_edge_features, edge_features + rev_edge_features],
                dim=-1,
            )

        # if self.bias_ln is not None:
        #     node_features = self.bias_ln(node_features)
        # if self.weight_ln is not None:
        #     edge_features = self.weight_ln(edge_features)

        node_features = self.proj_bias(node_features)
        edge_features = self.proj_weight(edge_features)

        if self.zero_out_weights:
            edge_features = torch.zeros_like(edge_features)
        if self.zero_out_bias:
            # only zero out bias, not gpf
            node_features = torch.zeros_like(node_features)

        if self.gpf is not None:
            probe_features = self.gpf(*inputs)
            # probe_features = self.proj_gpf(probe_features)
            node_features = node_features + probe_features

        node_features = self.proj_node_in(node_features)
        edge_features = self.proj_edge_in(edge_features)

        if self.use_pos_embed:
            pos_embed = torch.cat(
                [
                    # repeat(self.pos_embed[i], "d -> 1 n d", n=n)
                    self.pos_embed[i].unsqueeze(0).expand(1, n, -1)
                    for i, n in enumerate(self.pos_embed_layout)
                ],
                dim=1,
            )
            # print('pos_embed', pos_embed.shape, 'node_features', node_features.shape)
            # pos_embed torch.Size([1, 156, 64]) node_features torch.Size([1, 539, 64])
            node_features = node_features + pos_embed
        return node_features, edge_features, mask


class RTTransformer(nn.Module):
    def __init__(
        self,
        d_in,
        d_hid,
        d_out,
        n_layers,
        n_heads,
        layer_layout,
        dropout=0.0,
        node_update_type="rt",
        disable_edge_updates=False,
        use_cls_token=True,
        graph_features="mean",
        rev_edge_features=False,
        num_probe_features=0,
        zero_out_bias=False,
        zero_out_weights=False,
        bias_ln=False,
        weight_ln=False,
        sin_emb=False,
        input_layers=1,
        use_pos_embed=True,
        wave_pos_embed=False,
        use_topomask=False,
        inp_factor=1.0,
        normalize=False,
        modulate_v=True,
    ):
        super().__init__()
        # assert use_cls_token == (graph_features == "cls_token")
        self.graph_features = graph_features
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
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
            use_pos_embed=use_pos_embed,
            wave_pos_embed=wave_pos_embed,
            input_layers=input_layers,
            inp_factor=inp_factor,
            normalize=normalize,
        )
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(d_hid))

        self.layers = nn.ModuleList(
            [
                torch.jit.script(
                    RTLayer(
                        d_hid,
                        n_heads,
                        dropout,
                        node_update_type=node_update_type,
                        disable_edge_updates=disable_edge_updates,
                        use_topomask=use_topomask,
                        modulate_v=modulate_v,
                    )
                )
                for _ in range(n_layers)
            ]
        )
        num_graph_features = (
            layer_layout[-1] * d_hid if graph_features == "cat_last_layer" else d_hid
        )
        self.proj_out = nn.Sequential(
            nn.Linear(num_graph_features, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )
        
    def forward(self, inputs):
        node_features, edge_features, mask = self.construct_graph(inputs)
        # print('node_features', node_features.shape, 'edge_features', edge_features.shape, 'mask', mask.shape)
        # node_features torch.Size([8, 67, 64]) edge_features torch.Size([8, 67, 67, 64]) mask torch.Size([8, 67, 67, 1])

        if self.use_cls_token:
            node_features = torch.cat(
                [
                    # repeat(self.cls_token, "d -> b 1 d", b=node_features.size(0)),
                    self.cls_token.unsqueeze(0).expand(node_features.size(0), 1, -1),
                    node_features,
                ],
                dim=1,
            )
            edge_features = F.pad(edge_features, (0, 0, 1, 0, 1, 0), value=0)

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, mask)

        if self.graph_features == 'weights':
            new_w, new_b = graphs_to_batch(self.proj_out(edge_features),
                                           self.proj_out(node_features),
                                           *inputs)
            # print('new_w', len(new_w), new_w[0].shape, new_w[1].shape, new_w[2].shape,
            #       'new_b', len(new_b), new_b[0].shape, new_b[1].shape, new_b[2].shape)
            # new_w 3 torch.Size([8, 2, 32, 1]) torch.Size([8, 32, 32, 1]) torch.Size([8, 32, 1, 1])
            # new_b 3 torch.Size([8, 32, 1]) torch.Size([8, 32, 1]) torch.Size([8, 1, 1])
            x = (new_w, new_b)
        else:
            if self.graph_features == "cls_token":
                graph_features = node_features[:, 0]
            elif self.graph_features == "mean":
                graph_features = node_features.mean(dim=1)
            elif self.graph_features == "max":
                graph_features = node_features.max(dim=1).values
            elif self.graph_features == "last_layer":
                graph_features = node_features[:, -self.nodes_per_layer[-1] :].mean(dim=1)
            elif self.graph_features == "cat_last_layer":
                graph_features = node_features[:, -self.nodes_per_layer[-1] :].flatten(1, 2)
            elif self.graph_features == "mean_edge":
                graph_features = edge_features.mean(dim=(1, 2))
            elif self.graph_features == "max_edge":
                graph_features = edge_features.flatten(1, 2).max(dim=1).values
            elif self.graph_features == "last_layer_edge":
                graph_features = edge_features[:, -self.nodes_per_layer[-1] :, :].mean(
                    dim=(1, 2)
                )

            x = self.proj_out(graph_features)
            # print('x', x.shape)
            # x torch.Size([8, 10])

        return x
    

class RTransformerParams(nn.Module):
    def __init__(
        self,
        d_in,
        d_hid,
        d_out,
        n_layers,
        n_heads,
        layer_layout,
        dropout=0.0,
        node_update_type="rt",
        disable_edge_updates=False,
        rev_edge_features=False,
        num_probe_features=0,
        zero_out_bias=False,
        zero_out_weights=False,
        bias_ln=False,
        weight_ln=False,
        sin_emb=False,
        input_layers=1,
        use_pos_embed=True,
        wave_pos_embed=False,
        use_topomask=False,
        inp_factor=1.0,
        normalize=False,
        out_scale=0.01,
        modulate_v=True,
    ):
        super().__init__()
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
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
            use_pos_embed=use_pos_embed,
            wave_pos_embed=wave_pos_embed,
            input_layers=input_layers,
            inp_factor=inp_factor,
            normalize=normalize,
        )

        self.layers = nn.ModuleList(
            [
                torch.jit.script(
                    RTLayer(
                        d_hid,
                        n_heads,
                        dropout,
                        node_update_type=node_update_type,
                        disable_edge_updates=disable_edge_updates,
                        use_topomask=use_topomask,
                        modulate_v=modulate_v,
                    )
                )
                for _ in range(n_layers)
            ]
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

        self.weight_scale = nn.ParameterList([
            nn.Parameter(torch.tensor(out_scale)) for _ in range(len(layer_layout)-1)
        ])
        self.bias_scale = nn.ParameterList([
            nn.Parameter(torch.tensor(out_scale)) for _ in range(len(layer_layout)-1)
        ])

    def forward(self, inputs):
        node_features, edge_features, mask = self.construct_graph(inputs)

        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, mask)

        node_features = self.proj_node(node_features)
        edge_features = self.proj_edge(edge_features)

        weights, biases = graphs_to_batch(edge_features, node_features, *inputs)
        weights = [w * s for w, s in zip(weights, self.weight_scale)]
        biases = [b * s for b, s in zip(biases, self.bias_scale)]
        return weights, biases

        

class RTLayer(nn.Module):
    def __init__(
        self,
        d_hid,
        n_heads,
        dropout,
        node_update_type="rt",
        disable_edge_updates=False,
        use_topomask=False,
        modulate_v=True,
    ):
        super().__init__()
        self.node_update_type = node_update_type
        self.disable_edge_updates = disable_edge_updates

        self.self_attn = torch.jit.script(RTAttention(d_hid, d_hid, d_hid, n_heads, use_topomask=use_topomask, modulate_v=modulate_v))
        # self.self_attn = RTAttention(d_hid, d_hid, d_hid, n_heads)
        self.lin0 = nn.Linear(d_hid, d_hid)
        self.dropout0 = nn.Dropout(dropout)
        self.node_ln0 = nn.LayerNorm(d_hid)
        self.node_ln1 = nn.LayerNorm(d_hid)
        # if node_update_type == "norm_first":
        #     self.edge_ln0 = nn.LayerNorm(d_hid)

        act_fn = nn.GELU

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hid, 2 * d_hid, bias=False),
            act_fn(),
            nn.Linear(2 * d_hid, d_hid),
            nn.Dropout(dropout),
        )
        # self.ln1 = nn.LayerNorm(d_hid)

        if not self.disable_edge_updates:
            # self.reverse_edge = Rearrange("b n m d -> b m n d")
            # self.edge_mlp0 = nn.Sequential(
            #     nn.Linear(4 * d_hid, d_hid, bias=False),
            #     act_fn(),
            #     nn.Linear(d_hid, d_hid),
            #     nn.Dropout(dropout),
            # )
            self.edge_mlp0 = EdgeMLP(d_hid, act_fn, dropout)
            self.edge_mlp1 = nn.Sequential(
                nn.Linear(d_hid, 2 * d_hid, bias=False),
                act_fn(),
                nn.Linear(2 * d_hid, d_hid),
                nn.Dropout(dropout),
            )
            self.eln0 = nn.LayerNorm(d_hid)
            self.eln1 = nn.LayerNorm(d_hid)
            # self.eln2 = nn.LayerNorm(d_hid)
            # self.edge_attn = EfficientSelfAttention(d_hid=d_hid,
            #                                         d_out=d_hid,
            #                                         n_heads=n_heads, 
            #                                         dropout=dropout, 
            #                                         seq_len=4489)
            # self.elin = nn.Linear(4 * d_hid, 4 * d_hid)
            # TODO seq_len calculate from nodes per layer

    def node_updates(self, node_features, edge_features, mask):
        if self.node_update_type == "norm_first":
            node_features = node_features + self.self_attn(
                self.node_ln0(node_features), edge_features, mask
            )
            node_features = node_features + self.node_mlp(self.node_ln1(node_features))
        elif self.node_update_type == "norm_last":
            node_features = self.node_ln0(
                node_features + self.self_attn(node_features, edge_features, mask)
            )
            node_features = self.node_ln1(node_features + self.node_mlp(node_features))
        elif self.node_update_type == "rt":
            # attn_out = checkpoint(self.self_attn, node_features, edge_features, mask)
            node_features = self.node_ln0(
                node_features
                + self.dropout0(
                    self.lin0(self.self_attn(node_features, edge_features, mask))
                )
            )
            node_features = self.node_ln1(node_features + self.node_mlp(node_features))
        else:
            raise ValueError(f"Unknown node update type: {self.node_update_type}")
        return node_features

    def edge_updates(self, node_features, edge_features, mask):
        # source_nodes = repeat(
        #     node_features, "b n d -> b n m d", m=node_features.shape[1]
        # )
        # source_nodes = node_features.unsqueeze(-2).expand(
        #     -1, -1, node_features.size(-2), -1
        # )
        # target_nodes = repeat(
        #     node_features, "b n d -> b m n d", m=node_features.shape[1]
        # )
        # target_nodes = node_features.unsqueeze(-3).expand(
        #     -1, node_features.size(-2), -1, -1
        # )
        # reversed_edge_features = rearrange(edge_features, "b n m d -> b m n d")
        # reversed_edge_features = self.reverse_edge(edge_features)
        # input_features = torch.cat(
        #     [edge_features, reversed_edge_features, source_nodes, target_nodes],
        #     dim=-1,
        # )
        # input_features = self.elin(input_features).flatten(1,2)
        edge_features = self.eln0(edge_features + self.edge_mlp0(node_features, edge_features))
        # edge_features = self.eln2(edge_features + self.edge_attn(edge_features.flatten(1,2)).reshape(edge_features.shape))

        edge_features = self.eln1(edge_features + self.edge_mlp1(edge_features))
        return edge_features

    def forward(self, node_features, edge_features, mask):
        node_features = self.node_updates(node_features, edge_features, mask)

        if not self.disable_edge_updates:
            edge_features = self.edge_updates(node_features, edge_features, mask)

        return node_features, edge_features


class EdgeMLP(nn.Module):
    def __init__(self, d_hid, act_fn, dropout):
        super().__init__()
        self.d_hid = d_hid
        self.reverse_edge = Rearrange("b n m d -> b m n d")
        self.lin0_e = nn.Linear(2 * d_hid, d_hid)
        self.lin0_s = nn.Linear(d_hid, d_hid)
        self.lin0_t = nn.Linear(d_hid, d_hid)
        self.lin0_er = nn.Linear(d_hid, d_hid, bias=False)
        self.lin0_ec = nn.Linear(d_hid, d_hid, bias=False)
        self.act = act_fn()
        self.lin1 = nn.Linear(d_hid, d_hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_features, edge_features):
        source_nodes = self.lin0_s(node_features).unsqueeze(-2).expand(
            -1, -1, node_features.size(-2), -1
        )
        target_nodes = self.lin0_t(node_features).unsqueeze(-3).expand(
            -1, node_features.size(-2), -1, -1
        )
        source_edge = self.lin0_ec(edge_features.mean(dim=-3, keepdim=True)).expand(
            -1, edge_features.size(-3), -1, -1
        )
        target_edge = self.lin0_er(edge_features.mean(dim=-2, keepdim=True)).expand(
            -1, -1, edge_features.size(-2), -1
        )

        # reversed_edge_features = self.reverse_edge(edge_features)
        edge_features = self.lin0_e(torch.cat([edge_features, self.reverse_edge(edge_features)], dim=-1))
        edge_features = edge_features + source_nodes + target_nodes + source_edge + target_edge
        edge_features = self.act(edge_features)
        edge_features = self.lin1(edge_features)
        edge_features = self.drop(edge_features)

        return edge_features


class RTAttention(nn.Module):
    def __init__(self, d_node, d_edge, d_hid, n_heads, use_topomask=False, modulate_v=None):
        super().__init__()
        # assert n_heads % 4 == 0, "necessary for topoformer"

        self.n_heads = n_heads
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_hid = d_hid
        # self.use_topomask = use_topomask
        self.modulate_v = modulate_v
        self.scale = 1 / (d_hid**0.5)
        self.split_head_node = Rearrange("b n (h d) -> b h n d", h=n_heads)
        self.split_head_edge = Rearrange("b n m (h d) -> b h n m d", h=n_heads)
        self.cat_head_node = Rearrange("... h n d -> ... n (h d)", h=n_heads)

        self.qkv_node = nn.Linear(d_node, 3 * d_hid, bias=False)
        self.edge_factor = 4 if modulate_v else 3
        self.qkv_edge = nn.Linear(d_edge, self.edge_factor * d_hid, bias=False)
        # self.ln_q = nn.LayerNorm(d_hid // n_heads)
        # self.ln_k = nn.LayerNorm(d_hid // n_heads)

    def forward(self, node_features, edge_features, mask):
        qkv_node = self.qkv_node(node_features)
        # qkv_node = rearrange(qkv_node, "b n (h d) -> b h n d", h=self.n_heads)
        qkv_node = self.split_head_node(qkv_node)
        q_node, k_node, v_node = torch.chunk(qkv_node, 3, dim=-1)

        qkv_edge = self.qkv_edge(edge_features)
        # qkv_edge = rearrange(qkv_edge, "b n m (h d) -> b h n m d", h=self.n_heads)
        qkv_edge = self.split_head_edge(qkv_edge)
        qkv_edge = torch.chunk(qkv_edge, self.edge_factor, dim=-1)
        # q_edge, k_edge, v_edge, q_edge_b, k_edge_b, v_edge_b = torch.chunk(
        #     qkv_edge, 6, dim=-1
        # )

        q = q_node.unsqueeze(-2) + qkv_edge[0] # + q_edge_b
        k = k_node.unsqueeze(-3) + qkv_edge[1] # + k_edge_b
        if self.modulate_v:
            v = v_node.unsqueeze(-3) * qkv_edge[3] + qkv_edge[2]
        else:
            v = v_node.unsqueeze(-3) + qkv_edge[2]

        # q = self.ln_q(q)
        # k = self.ln_k(k)
        dots = self.scale * torch.einsum("b h i j d, b h i j d -> b h i j", q, k)

        attn = F.softmax(dots, dim=-1)
        out = torch.einsum("b h i j, b h i j d -> b h i d", attn, v)
        # out = rearrange(out, "b h n d -> b n (h d)")
        out = self.cat_head_node(out)
        return out


class EfficientSelfAttention(nn.Module):
    def __init__(self, n_heads, d_hid, d_out, **kwargs) -> None:
        super().__init__()
        # self.attn = attention.LinformerAttention(**kwargs)
        # self.n_heads = n_heads
        # self.split_head = Rearrange("b n (h d) -> b h n d", h=n_heads)
        # self.cat_head = Rearrange("b h n d -> b n (h d)")

        # self.mhd = MultiHeadDispatch(d_hid, n_heads, self.attn)
        self.mha = nn.MultiheadAttention(d_hid, n_heads, dropout=kwargs["dropout"])
        self.proj_out = nn.Linear(d_hid, d_out)

    def forward(self, x):
        # x = self.split_head(x)
        # x = self.attn(x, x, x)
        # x = self.cat_head(x)
        x, _ = checkpoint(self.mha, (x, x, x), (), True)
        x = self.proj_out(x)
        return x
    

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads