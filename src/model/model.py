import os

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from omegaconf import DictConfig, OmegaConf

from src.data.transform import AddNodeFeatures
from src.model.modules import GNNLayer, FeatureEncoder, SinusoidalNumericalEncoder


class GNN(nn.Module):

    def __init__(
            self,
            channels: int,
            feat_dim: int,
            num_layers: int,
            out_dim: int = 2,
            aggr: str | list[str] = "mean",
            feature_encoder: str = "mlp",
            dropout: float = 0.0,
            var_output: bool = True,
    ):
        """
        A message-passing Graph Neural Network
        :param channels: Hidden model dimension
        :param feat_dim: Dimension of input node features
        :param num_layers: Number of message passing layers
        :param out_dim: node-level output dimension
        :param aggr: Message aggregation function either mean, max, or sum. If a list is provided, than multiple types of aggregation are performed in parallel.
        :param feature_encoder: Type of node feature encoder. Either "mlp" for a simple perceptron or "sin" for a sinusoidal numerical encoder.
        :param dropout: Dropout probability
        :param var_output: If true, the output will be per variable. If false, the output will be per literal, which is useful for our supervised tasks like backbone prediction.
        """
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
            GNNLayer(channels=channels, aggr=aggr, dropout=dropout) for _ in range(num_layers)
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
            # concatenate interleaved embeddings and apply an mlp
            h_var = torch.cat([h_lit[0::2], h_lit[1::2]], dim=1)
            y_var = self.out_lin2(self.out_act(self.out_lin1(h_var)))
            return y_var
        else:
            y_lit = self.out_lin2(self.out_act(self.out_lin1(h_lit)))
            return y_lit


def init_model(cfg: DictConfig, transform: AddNodeFeatures, **model_kwargs) -> GNN:
    model = GNN(
        channels=cfg.model.channels,
        feat_dim=transform.lit_dim(),
        num_layers=cfg.model.num_layers,
        aggr=OmegaConf.to_container(cfg.model.aggr),
        feature_encoder=cfg.model.feature_encoder,
        dropout=cfg.model.dropout if "dropout" in cfg.model else 0.0,
        **model_kwargs,
    )
    return model


def load_checkpoint(ckpt_path: str, **model_kwargs) -> tuple[GNN, AddNodeFeatures, DictConfig]:
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    transform = AddNodeFeatures()

    model = init_model(cfg, transform, **model_kwargs)

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    return model, transform, cfg
