import os
import hydra
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

import wandb
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from torchmetrics.functional import auroc, accuracy


from src.data.dataset import LabeledDataset
from src.model.model import GNN
from src.data.transform import AddNodeFeatures

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


@torch.no_grad()
def evaluate(
        model: GNN,
        loader: DataLoader,
        device: torch.device | str = "cpu",
        split: str = "val",
) -> dict[str, float]:
    model.to(device)
    model.eval()

    target_all = []
    y_pred_all = []

    for data in loader:
        data.to(device)

        y_pred = model(data).flatten().cpu()
        target = data["lit"].target.cpu()

        target_all.append(target)
        y_pred_all.append(y_pred)

    target = torch.cat(target_all, dim=0)
    y_pred = torch.cat(y_pred_all, dim=0)
    loss = F.binary_cross_entropy_with_logits(y_pred, target).item()

    metrics = {
        f"{split}/loss": loss,
        f"{split}/accuracy": accuracy(y_pred, target.int(), "binary"),
        f"{split}/roc_auc": auroc(y_pred, target.int(), "binary"),
    }
    return metrics


def train(
        model: GNN,
        cfg: DictConfig,
        loader_train: DataLoader,
        loader_val: DataLoader,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        epochs: int,
        global_step: int = 0,
        device: torch.device | str = "cpu",
        use_amp: bool = True,
) -> int:
    scaler = torch.amp.GradScaler() if use_amp else None
    model.to(device)
    best_loss = np.inf

    loss_all = []
    for e in range(1, epochs + 1):
        model.train()
        for data in tqdm(loader_train, desc=f"Epoch {e}"):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                optim.zero_grad()
                data.to(device)

                y_pred = model(data).flatten()
                target = data["lit"].target

                loss = F.binary_cross_entropy_with_logits(y_pred, target)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

                sched.step()

                loss_all.append(loss.item())
                global_step += 1

        train_metrics = {
            "train/loss": np.mean(loss_all),
            "train/lr": sched.get_last_lr()[0]
        }
        wandb.log(train_metrics, step=global_step)
        loss_all = []

        val_metrics = evaluate(model, loader_val, device)
        wandb.log(val_metrics, step=global_step)

        if val_metrics["val/loss"] < best_loss:
            best_loss = val_metrics["val/loss"]
            save_model(model, cfg, "best")

    return global_step


def save_model(model: GNN, cfg: DictConfig, checkpoint_name: str = "last") -> None:
    model_dir = cfg.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    cfg_path = os.path.join(model_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        OmegaConf.save(cfg, f)
    ckpt_path = os.path.join(model_dir, f"{checkpoint_name}.pt")
    state_dict = model.state_dict()
    torch.save(state_dict, ckpt_path)


def load_checkpoint(ckpt_path: str) -> tuple[GNN, AddNodeFeatures]:
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    transform = AddNodeFeatures()

    model = GNN(
        channels=cfg.model.channels,
        feat_dim=transform.lit_dim(),
        num_layers=cfg.model.num_layers,
        aggr=OmegaConf.to_container(cfg.model.aggr),
        var_output=False,
    )

    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    return model, transform


@hydra.main(version_base=None, config_path="configs", config_name="config_train_supervised")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg),
    )

    if cfg.from_checkpoint is not None:
        model, transform = load_checkpoint(cfg.from_checkpoint)
    else:
        transform = AddNodeFeatures()

        model = GNN(
            channels=cfg.model.channels,
            feat_dim=transform.lit_dim(),
            num_layers=cfg.model.num_layers,
            aggr=OmegaConf.to_container(cfg.model.aggr),
            var_output=False,
            out_dim=1,
        )

    dataset_train = LabeledDataset(
        path=cfg.dataset.train_path,
        transform=transform,
        target=cfg.dataset.target,
        num_workers=cfg.dataset.num_workers,
    )
    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=True,
    )

    dataset_val = LabeledDataset(
        path=cfg.dataset.val_path,
        transform=transform,
        target=cfg.dataset.target,
        num_workers=cfg.dataset.num_workers,
    )
    loader_val = DataLoader(
        dataset=dataset_val,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    global_step = 0

    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    T_max = cfg.training.epochs * len(loader_train)
    warmup_steps = int(T_max * cfg.optim.warmup_ratio)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer=optim,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=T_max - warmup_steps,
        eta_min=cfg.optim.lr_min,
    )
    sched = torch.optim.lr_scheduler.SequentialLR(optim, [warmup, cosine], [warmup_steps])

    train(
        model=model,
        cfg=cfg,
        optim=optim,
        sched=sched,
        loader_train=loader_train,
        loader_val=loader_val,
        epochs=cfg.training.epochs,
        global_step=global_step,
        device=device,
    )

    wandb.finish()


if __name__ == '__main__':
    main()
