import math

import numpy as np
import torch
import torch_sparse
from torch import Tensor, nn as nn
from torch_geometric.data import HeteroData
from torch_scatter import scatter
from torch_sparse import SparseTensor, cat


def flip_lit_emb(x_lit: Tensor) -> Tensor:
    x_flip = torch.empty_like(x_lit)
    x_flip[0::2] = x_lit[1::2]
    x_flip[1::2] = x_lit[0::2]
    return x_flip


def aggregate(adj: SparseTensor, x: Tensor, aggr: str | list[str]):
    if isinstance(aggr, str):
        return torch_sparse.matmul(adj, x, reduce=aggr)
    else:
        assert x.shape[1] % len(aggr) == 0
        split_size = x.shape[1] // len(aggr)
        result = []
        for x_split, aggr_split in zip(torch.split(x, split_size, dim=1), aggr):
            y_split = torch_sparse.matmul(adj, x_split, reduce=aggr_split)
            result.append(y_split)
        return torch.cat(result, dim=1)


class GNNLayer(nn.Module):

    def __init__(self, channels: int, aggr: str | list[str], global_aggr: str | None = None, dropout: float = 0.0):
        super(GNNLayer, self).__init__()
        self.channels = channels
        self.global_aggr = global_aggr
        self.dropout = dropout

        if isinstance(aggr, str):
            aggr = [aggr]
        self.aggr = aggr

        self.lit_norm = nn.LayerNorm(channels)
        self.cls_norm = nn.LayerNorm(channels)

        lit_dim_in = 3 * channels
        if global_aggr is not None:
            lit_dim_in += channels

        self.lit_up = nn.Sequential(
            nn.Linear(lit_dim_in, channels),
            nn.SiLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(channels, channels),
        )
        self.cls_up = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.SiLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(channels, channels),
        )

    def forward(self, h_lit, h_cls, data: HeteroData) -> tuple[Tensor, Tensor]:
        if "cls_adj" in data["cls", "lit"]:
            cls_adj = data["cls", "lit"].cls_adj
        else:
            edge_index = data["cls", "lit"].edge_index
            size = (data["cls"].num_nodes, data["lit"].num_nodes)
            cls_adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=size)
            data["cls", "lit"].cls_adj = cls_adj

        if "lit_adj" in data["cls", "lit"]:
            lit_adj = data["cls", "lit"].lit_adj
        else:
            lit_adj = cls_adj.t()
            data["cls", "lit"].lit_adj = lit_adj

        z_cls = [h_cls, aggregate(cls_adj, h_lit, self.aggr)]
        z_cls = torch.cat(z_cls, dim=1)
        h_cls = h_cls + self.cls_up(z_cls)
        h_cls = self.cls_norm(h_cls)

        z_lit = [h_lit, flip_lit_emb(h_lit), aggregate(lit_adj, h_cls, self.aggr)]
        if self.global_aggr is not None:
            lit_batch = data["lit"].batch
            x_glob = scatter(h_lit, lit_batch, dim=0, reduce=self.global_aggr)
            z_lit.append(x_glob[lit_batch])

        z_lit = torch.cat(z_lit, dim=1)
        h_lit = h_lit + self.lit_up(z_lit)
        h_lit = self.lit_norm(h_lit)

        return h_lit, h_cls


class ETLayer(nn.Module):

    def __init__(self, channels: int, nheads: int = 8):
        super(ETLayer, self).__init__()
        self.channels = channels
        self.nheads = nheads

        assert channels % nheads == 0
        self.head_dim = channels // nheads
        self.a1lin = nn.Linear(channels, nheads, bias=False)
        self.a2lin = nn.Linear(channels, nheads, bias=False)
        self.v1lin = nn.Linear(channels, channels, bias=False)
        self.v2lin = nn.Linear(channels, channels, bias=False)
        self.olin = nn.Linear(channels, channels, bias=False)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.up = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def _tri_attn(self, h: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        num_batches = h.size(0)
        num_nodes_q = h.size(1)
        num_nodes_k = h.size(1)

        # project embeddings directly to pairwise attention scores
        x = h
        a1 = self.a1lin(x)  # (b,i,l,h)
        a2 = self.a2lin(x)  # (b,l,j,h)
        # A_{i,l,j} = a1_{i,l} + a2_{l,j}

        # subtract max for stability and apply masks before exp
        a1 -= a1.max(dim=2, keepdim=True)[0]
        a2 -= a2.max(dim=1, keepdim=True)[0]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(-1)
            a1.masked_fill_(attn_mask, float("-inf"))
            a2.masked_fill_(attn_mask, float("-inf"))
        a1.exp_()
        a2.exp_()

        # get value embeddings and reshape
        v1 = self.v1lin(x)
        v2 = self.v2lin(x)
        v1 = v1.view(
            num_batches, num_nodes_q, num_nodes_q, self.nheads, self.head_dim  # (b,i,l,h,d)
        )
        v2 = v2.view(
            num_batches, num_nodes_k, num_nodes_k, self.nheads, self.head_dim  # (b,l,j,h,d)
        )

        # multiply with unnormalized, exponentiated attention weights
        v1.mul_(a1.unsqueeze(-1))
        v2.mul_(a2.unsqueeze(-1))

        # Aggregate (PPGN style)
        out = torch.einsum("bilhd,bljhd->bijhd", v1, v2)

        # Get attention denominator and normalize (lazy softmax)
        denom = torch.einsum("bilh,bljh->bijh", a1, a2)
        denom.add_(1e-6)
        out.div_(denom.unsqueeze(-1))

        out = out.view(num_batches, num_nodes_q, num_nodes_k, self.channels)
        return self.olin(out)

    @torch.compile
    def forward(self, h: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.norm1(h + self._tri_attn(h, attn_mask))
        h = self.norm2(h + self.up(h))
        return h


class SinusoidalNumericalEncoder(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, gamma: float = 1.0, dropout: float =0.0):
        super(SinusoidalNumericalEncoder, self).__init__()
        assert channels_out // 2

        self.scale = 1 / np.sqrt(channels_out / 2)
        self.lin = nn.Linear(channels_in, channels_out // 2, bias=False)
        torch.nn.init.normal_(self.lin.weight, 0.0, gamma**-2)

        self.mlp = nn.Sequential(
            nn.Linear(channels_out, channels_out),
            nn.SiLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(channels_out, channels_out),
            nn.LayerNorm(channels_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x.mul_(self.scale)
        return self.mlp(x)


class FeatureEncoder(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, dropout: float = 0.0):
        super(FeatureEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels_in, 2 * channels_out),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(2 * channels_out, channels_out),
            nn.LayerNorm(channels_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
