import numpy as np
import torch
import torch_sparse
from torch import Tensor, nn as nn
from torch_geometric.data import HeteroData
from torch_sparse import SparseTensor


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

    def __init__(self, channels: int, aggr: str | list[str], dropout: float = 0.0):
        super(GNNLayer, self).__init__()
        self.channels = channels
        self.dropout = dropout

        if isinstance(aggr, str):
            aggr = [aggr]
        self.aggr = aggr

        self.lit_norm = nn.LayerNorm(channels)
        self.cls_norm = nn.LayerNorm(channels)

        lit_dim_in = 3 * channels
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
        """
        :param h_lit:
        :param h_cls:
        :param data:
        :return:
        """
        # Convert literal -> clause adjacency matrix to a SparseTensor for memory efficient aggregation.
        # The SparseTensor is cached for the following layers.
        if "cls_adj" in data["cls", "lit"]:
            cls_adj = data["cls", "lit"].cls_adj
        else:
            edge_index = data["cls", "lit"].edge_index
            size = (data["cls"].num_nodes, data["lit"].num_nodes)
            cls_adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=size)
            data["cls", "lit"].cls_adj = cls_adj

        # literal -> clause message pass
        z_cls = [h_cls, aggregate(cls_adj, h_lit, self.aggr)]
        z_cls = torch.cat(z_cls, dim=1)
        h_cls = h_cls + self.cls_up(z_cls)
        h_cls = self.cls_norm(h_cls)

        # Cache clause -> literal adjacency matrix as SparseTensor.
        if "lit_adj" in data["cls", "lit"]:
            lit_adj = data["cls", "lit"].lit_adj
        else:
            lit_adj = cls_adj.t()
            data["cls", "lit"].lit_adj = lit_adj

        # clause -> literal message pass
        z_lit = [h_lit, flip_lit_emb(h_lit), aggregate(lit_adj, h_cls, self.aggr)]
        z_lit = torch.cat(z_lit, dim=1)
        h_lit = h_lit + self.lit_up(z_lit)
        h_lit = self.lit_norm(h_lit)

        return h_lit, h_cls


class SinusoidalNumericalEncoder(nn.Module):
    # Sinusoidal node feature encoder
    def __init__(self, channels_in: int, channels_out: int, gamma: float = 1.0, dropout: float = 0.0):
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
    # Simple MLP-based node feature encoder
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
