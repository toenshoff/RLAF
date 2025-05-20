import torch
from torch import Tensor
from torch_geometric.utils import degree


def drop_zeros_and_force_ones(mat: Tensor) -> Tensor:
    assert mat.layout == torch.sparse_coo
    indices = mat.indices()
    values = mat.values()
    mask = values != 0
    values = torch.ones_like(values[mask])
    return torch.sparse_coo_tensor(indices=indices[:, mask], values=values, size=mat.size())


def row_deg_coo(mat: Tensor) -> Tensor:
    assert mat.layout == torch.sparse_coo
    row, _ = mat.coalesce().indices()
    deg = degree(row, num_nodes=mat.shape[0], dtype=torch.int32)
    return deg
