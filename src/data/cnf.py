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


@torch.compiler.disable(recursive=False)
def to_dense_var_var_feat(data: HeteroData, rrwp_steps: int = -1) -> tuple[Tensor, Tensor, Tensor]:
    device = data["cls", "lit"].edge_index.device
    if "batch" in data["lit"]:
        batch = data["lit"].batch
    else:
        batch = torch.zeros((data["lit"].num_nodes,), dtype=torch.long, device=device)

    edge_index = data["cls", "lit"].edge_index
    size = (data["cls"].num_nodes, data["lit"].num_nodes)
    A = to_torch_coo_tensor(edge_index, size=size)
    A = A.t() @ A
    A_ll = to_dense_adj(A.indices(), batch=batch, edge_attr=A.values())
    batch_size, num_lit, _ = A_ll.shape

    num_var = num_lit // 2

    if rrwp_steps <= 0:
        x = torch.zeros((num_var, num_var, 5), device=device)
        idx = torch.arange(0, num_lit, 2, device=device, dtype=torch.long)
        idx_flip = flip_lit_idx(idx)
        x[:, :, 0].fill_diagonal_(1.0)

        x = x.unsqueeze(0).tile(batch_size, 1, 1, 1)
        x[:, :, :, 1] = A_ll[:, idx.unsqueeze(0), idx.unsqueeze(1)]
        x[:, :, :, 2] = A_ll[:, idx.unsqueeze(0), idx_flip.unsqueeze(1)]
        x[:, :, :, 3] = A_ll[:, idx_flip.unsqueeze(0), idx.unsqueeze(1)]
        x[:, :, :, 4] = A_ll[:, idx_flip.unsqueeze(0), idx_flip.unsqueeze(1)]
        x.add_(1)
        x.log_()
    else:
        P = A_ll.clone()
        #lit_idx = torch.arange(0, num_lit, device=device, dtype=torch.long)
        #P[:, lit_idx, lit_idx] = 0.0
        P /= P.sum(dim=-1, keepdim=True) + 1e-6

        WP = torch.zeros((batch_size, num_lit, num_lit, rrwp_steps), dtype=P.dtype, device=P.device)
        WP[:, :, :, 0] = P
        for s in range(1, rrwp_steps):
            WP[:, :, :, s] = P @ WP[:, :, :, s-1]

        D = (WP == 0.0).float().sum(dim=-1)
        #idx = torch.arange(0, num_lit, 2, device=device, dtype=torch.long)
        #idx_flip = flip_lit_idx(idx)
        #x = torch.cat([
        #    WP[:, idx.unsqueeze(0), idx.unsqueeze(1)],
        #    WP[:, idx.unsqueeze(0), idx_flip.unsqueeze(1)],
        #    WP[:, idx_flip.unsqueeze(0), idx.unsqueeze(1)],
        #    WP[:, idx_flip.unsqueeze(0), idx_flip.unsqueeze(1)],
        #], dim=-1)

        x = torch.zeros((num_var, num_var, 5), device=device)
        idx = torch.arange(0, num_lit, 2, device=device, dtype=torch.long)
        idx_flip = flip_lit_idx(idx)
        x[:, :, 0].fill_diagonal_(1.0)

        x = x.unsqueeze(0).tile(batch_size, 1, 1, 1)
        x[:, :, :, 1] = D[:, idx.unsqueeze(0), idx.unsqueeze(1)]
        x[:, :, :, 2] = D[:, idx.unsqueeze(0), idx_flip.unsqueeze(1)]
        x[:, :, :, 3] = D[:, idx_flip.unsqueeze(0), idx.unsqueeze(1)]
        x[:, :, :, 4] = D[:, idx_flip.unsqueeze(0), idx_flip.unsqueeze(1)]

    var_idx = torch.arange(num_var, device=x.device)
    num_lit_batched = degree(batch, num_nodes=batch_size, dtype=var_idx.dtype)
    padding_mask = num_lit_batched.unsqueeze(1) / 2 > var_idx.unsqueeze(0)

    attn_mask = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
    attn_mask = ~attn_mask

    return x, padding_mask, attn_mask
