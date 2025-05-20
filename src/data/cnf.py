import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree, coalesce, to_undirected, to_torch_coo_tensor, to_dense_adj
from src.utils import drop_zeros_and_force_ones


def to_normalized_lit(l: int):
    return 2 * (abs(l) - 1) + ((1 + np.sign(l)) // 2)


def to_signed_lit(l: int):
    return (l // 2) * (-1 + 2 * (l % 2)) + 1


def cnf_to_adj(f: list[list[int]], num_var: int | None = None) -> Tensor:
    num_cls = len(f)
    if num_var is None:
        num_lit = 2 * max(max(abs(l) for l in c) for c in f)
    else:
        num_lit = 2 * num_var

    lit_idx = np.concatenate(f, axis=0)
    arity = np.array([len(c) for c in f])
    cls_idx = np.repeat(np.arange(num_cls), arity, axis=0)

    # map signed literals to normalized literal idx
    lit_idx = 2 * (np.abs(lit_idx) - 1) + ((1 + np.sign(lit_idx)) // 2)

    cls_idx = torch.tensor(cls_idx)
    lit_idx = torch.tensor(lit_idx)
    indices = torch.stack([cls_idx, lit_idx], dim=0)
    values = torch.ones((indices.shape[1],), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=(num_cls, num_lit))
    adj = adj.coalesce()
    adj = drop_zeros_and_force_ones(adj)
    return adj


def adj_to_cnf(adj: Tensor) -> list[list[int]]:
    adj = adj.to_sparse_csr()

    # map literals back to signed format
    lit_idx = adj.col_indices()
    lit_idx = ((lit_idx // 2) + 1) * (-1 + 2 * (lit_idx % 2))

    row_ptr = adj.crow_indices()
    f = []
    for i in range(adj.shape[0]):
        low, high = row_ptr[i], row_ptr[i+1]
        f.append(lit_idx[low:high].tolist())
    return f


def cnf_to_pyg(f: list[list[int]], num_var: int | None = None):
    adj = cnf_to_adj(f, num_var=num_var)

    data = HeteroData()
    data["cls"].num_nodes = adj.shape[0]
    data["lit"].num_nodes = adj.shape[1]

    edge_index = adj.coalesce().indices()
    data["cls", "lit"].edge_index = edge_index
    return data


def flip_lit_idx(idx: Tensor) -> Tensor:
    return idx + (1 - 2 * (idx % 2))


def flip_lit_columns(adj: Tensor):
    assert adj.layout == torch.sparse_coo
    adj = adj.coalesce()
    indices = adj.indices().clone()
    indices[1].add_(1 - 2 * (indices[1] % 2))
    return torch.sparse_coo_tensor(indices=indices, values=adj.values(), size=adj.size())
