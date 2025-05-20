import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from torch_geometric.utils import degree, to_undirected
from torch_geometric.transforms import BaseTransform


def get_rwpe(data: HeteroData, walk_length: int) -> dict[str, Tensor]:
        node_types = list(data.node_types)
        hom = data.to_homogeneous(add_node_type=True)

        hom.edge_index = to_undirected(hom.edge_index)
        edge_index, num_nodes = hom.edge_index, hom.num_nodes

        adj: torch.Tensor = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1

        deg_inv = 1 / (adj.sum(dim=1) + 1e-8)
        A = adj * deg_inv.view(-1, 1)
        P = A
        A = A.to_sparse(layout=torch.sparse_csr)

        probs = torch.empty((num_nodes, num_nodes, walk_length))
        for k in range(walk_length):
            probs[:, :, k] = P
            P = A @ P

        diag_idx = torch.arange(num_nodes, device=probs.device)
        pe = probs[diag_idx, diag_idx].contiguous()

        hom_node_types = hom.node_type
        pe_dict = {}
        for type_idx, nt in enumerate(node_types):
            mask = (hom_node_types == type_idx)
            pe_dict[nt] = pe[mask]

        return pe_dict


class AddNodeFeatures(BaseTransform):

    def __init__(self, rwpe_walk_length: int = -1):
        super(AddNodeFeatures, self).__init__()
        self.rwpe_walk_length = rwpe_walk_length

    def lit_dim(self) -> int:
        return 1 if self.rwpe_walk_length <= 0 else 1 + self.rwpe_walk_length

    def cls_dim(self) -> int:
        return 1 if self.rwpe_walk_length <= 0 else 1 + self.rwpe_walk_length

    def forward(self, data: HeteroData) -> HeteroData:
        edge_index = data["cls", "lit"].edge_index
        num_cls, num_lit = data["cls"].num_nodes, data["lit"].num_nodes

        cls_deg = degree(edge_index[0], num_cls, dtype=torch.float32).unsqueeze(1)
        lit_deg = degree(edge_index[1], num_lit, dtype=torch.float32).unsqueeze(1)

        if ("lit", "to", "lit") in data.edge_types and data["lit", "lit"].edge_index is not None:
            lit_deg += degree(data["lit", "lit"].edge_index[1], num_nodes=num_lit).unsqueeze(1)

        x_cls = torch.log(1 + cls_deg)
        x_lit = torch.log(1 + lit_deg)

        if self.rwpe_walk_length > 0:
            pe_dict = get_rwpe(data, walk_length=self.rwpe_walk_length)
            x_cls = torch.cat([x_cls, pe_dict["cls"]], dim=1)
            x_lit = torch.cat([x_lit, pe_dict["lit"]], dim=1)

        data["cls"].x = x_cls
        data["lit"].x = x_lit
        return data
