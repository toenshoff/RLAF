import os

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from torch import Tensor
from torch_geometric.data import HeteroData

from src.data.cnf import cnf_to_pyg, to_dense_var_var_feat
from src.data.transform import AddNodeFeatures
from src.model.modules import GNNLayer, ETLayer, FeatureEncoder, SinusoidalNumericalEncoder

from src.data.io_utils import load_dimacs_cnf


class GNN(nn.Module):

    def __init__(
            self,
            channels: int,
            feat_dim: int,
            num_layers: int,
            out_dim: int = 2,
            aggr: str | list[str] = "mean",
            global_aggr: str | list[str] | None = None,
            feature_encoder: str = "mlp",
            dropout: float = 0.0,
            var_output: bool = True,
    ):
        super(GNN, self).__init__()
        self.channels = channels

        if feature_encoder == "mlp":
            self.deg_enc = FeatureEncoder(
                channels_in=feat_dim,
                channels_out=channels,
                dropout=dropout,
            )
        elif feature_encoder == "sin":
            self.deg_enc = SinusoidalNumericalEncoder(
                channels_in=feat_dim,
                channels_out=channels,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown feature encoder type {feature_encoder}")

        self.layers = nn.ModuleList([
            GNNLayer(channels=channels, aggr=aggr, global_aggr=global_aggr, dropout=dropout) for _ in range(num_layers)
        ])

        self.var_output = var_output
        if self.var_output:
            # output mlp with last layer initialized with zeros
            self.out_lin1 = nn.Linear(2 * channels, 2 * channels)
            self.out_lin2 = nn.Linear(2 * channels, out_dim)
            nn.init.zeros_(self.out_lin2.weight)
            self.out_act = nn.SiLU(inplace=True)
        else:
            # output mlp with last layer initialized with zeros
            self.out_lin1 = nn.Linear(channels, 2 * channels)
            self.out_lin2 = nn.Linear(2 * channels, out_dim)
            self.out_act = nn.SiLU(inplace=True)

    def forward(self, data: HeteroData) -> Tensor:
        x_lit = data["lit"].x
        h_lit = self.deg_enc(x_lit)

        x_cls = data["cls"].x
        h_cls = self.deg_enc(x_cls)

        for layer in self.layers:
            h_lit, h_cls = layer(h_lit, h_cls, data)

        if self.var_output:
            h_var = torch.cat([h_lit[0::2], h_lit[1::2]], dim=1)
            y_var = self.out_lin2(self.out_act(self.out_lin1(h_var)))
            return y_var
        else:
            y_lit = self.out_lin2(self.out_act(self.out_lin1(h_lit)))
            return y_lit


class ET(nn.Module):

    def __init__(
            self,
            channels: int,
            num_layers: int,
            out_dim: int = 2,
            nheads: int = 8,
            rrwp_steps: int = -1,
            feature_encoder: str = "mlp",
    ):
        super(ET, self).__init__()
        self.channels = channels

        if feature_encoder == "mlp":
            self.in_proj = FeatureEncoder(
                channels_in=5,
                channels_out=channels,
            )
        elif feature_encoder == "sin":
            self.in_proj = SinusoidalNumericalEncoder(
                channels_in=5,
                channels_out=channels,
            )
        else:
            raise ValueError(f"Unknown feature encoder type {feature_encoder}")

        self.layers = nn.ModuleList([
            ETLayer(channels=channels, nheads=nheads) for _ in range(num_layers)
        ])

        # output mlp with last layer initialized with zeros
        self.out_lin1 = nn.Linear(channels, 2 * channels)
        self.out_lin2 = nn.Linear(2 * channels, out_dim)
        nn.init.zeros_(self.out_lin2.weight)
        self.out_act = nn.SiLU(inplace=True)

        self.rrwp_steps = rrwp_steps

    def forward(self, data: HeteroData) -> Tensor:
        x, padding_mask, attn_mask = to_dense_var_var_feat(data, rrwp_steps=self.rrwp_steps)
        h = self.in_proj(x)

        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)

        h_var = torch.diagonal(h, 0, 1, 2).transpose(1, 2)
        h_var = h_var[padding_mask]

        y_var = self.out_lin2(self.out_act(self.out_lin1(h_var)))
        return y_var


def init_model(cfg: DictConfig, transform: AddNodeFeatures) -> GNN | ET:
    model_type = cfg.model.type if "type" in cfg.model else "MPNN"

    if model_type == "MPNN":
        model = GNN(
            channels=cfg.model.channels,
            feat_dim=transform.lit_dim(),
            num_layers=cfg.model.num_layers,
            aggr=OmegaConf.to_container(cfg.model.aggr),
            feature_encoder=cfg.model.feature_encoder,
            dropout=cfg.model.dropout if "dropout" in cfg.model else 0.0,
        )
    elif model_type == "ET":
        model = ET(
            channels=cfg.model.channels,
            num_layers=cfg.model.num_layers,
            nheads=cfg.model.nheads,
            feature_encoder=cfg.model.feature_encoder,
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")

    return model


def load_checkpoint(ckpt_path: str) -> tuple[GNN, AddNodeFeatures, DictConfig]:
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    transform = AddNodeFeatures()

    model = init_model(cfg, transform)

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    return model, transform, cfg


if __name__ == '__main__':

    #f_1 = [[1, 2, 3], [4, -2], [6, -2], [-1, -4], [2, 5], [5, -3], [-1, -2], [-1, 2], [1, -2], [1, 2, -4]]
    f_1 = load_dimacs_cnf("../../data/test/SGen1/unsat/unsat_0.cnf")

    data = cnf_to_pyg(f_1).to("cuda")

    model = ET(
        channels=128,
        num_layers=4,
    )
    model.to("cuda")

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            y_lit = model(data)

            assignment = model.head.sample(y_lit, num_samples=1)
            print(assignment)
