import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, n_queries, d_hid, n_layers, n_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_hid, n_heads, dim_feedforward=2*d_hid, 
                                       dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.init_query = nn.Parameter(torch.randn(1, n_queries, d_hid))

    def forward(self, memory):
        tgt = self.init_query.repeat(memory.shape[0], 1, 1)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt