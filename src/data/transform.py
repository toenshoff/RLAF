import torch
from torch_geometric.data import HeteroData

from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform


class AddNodeFeatures(BaseTransform):

    def __init__(self):
        super(AddNodeFeatures, self).__init__()

    def lit_dim(self) -> int:
        return 1

    def cls_dim(self) -> int:
        return 1

    def forward(self, data: HeteroData) -> HeteroData:
        edge_index = data["cls", "lit"].edge_index
        num_cls, num_lit = data["cls"].num_nodes, data["lit"].num_nodes

        cls_deg = degree(edge_index[0], num_cls, dtype=torch.float32).unsqueeze(1)
        lit_deg = degree(edge_index[1], num_lit, dtype=torch.float32).unsqueeze(1)

        if ("lit", "to", "lit") in data.edge_types and data["lit", "lit"].edge_index is not None:
            lit_deg += degree(data["lit", "lit"].edge_index[1], num_nodes=num_lit).unsqueeze(1)

        x_cls = torch.log(1 + cls_deg)
        x_lit = torch.log(1 + lit_deg)

        data["cls"].x = x_cls
        data["lit"].x = x_lit
        return data
