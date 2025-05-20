import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader

import wandb
from src.model.model import GNN
import src.policy.policy as policy


def get_grpo_advantage(
        solver_stats: pd.DataFrame,
        target_stat: str = "decisions",
) -> np.ndarray:
    max_cost = solver_stats[target_stat].max()
    solver_stats[target_stat] = solver_stats[target_stat].fillna(max_cost)

    grouped = solver_stats[["cnf_id", target_stat]].groupby("cnf_id")

    mean = grouped.mean()
    target_mean = mean.loc[solver_stats["cnf_id"]]
    target_mean = target_mean[target_stat].to_numpy()

    std = grouped.std()
    target_std = std.loc[solver_stats["cnf_id"]]
    target_std = target_std[target_stat].to_numpy()

    target_val = solver_stats[target_stat].to_numpy()

    eps = 1e-8
    advantage = - (target_val - target_mean) / (target_std + eps)
    return advantage


def objective(
        log_prob: Tensor,
        log_prob_ref: Tensor,
        advantage: Tensor,
        clip_ratio: float = 0.2
) -> tuple[Tensor, Tensor]:
    g = advantage.clone()
    g[advantage >= 0.0] *= 1 + clip_ratio
    g[advantage < 0.0] *= 1 - clip_ratio

    prob_ratio = torch.exp(log_prob - log_prob_ref)
    L = torch.minimum(prob_ratio * advantage, g)

    return L.mean(), prob_ratio


def train_grpo(
        model: GNN,
        loader: DataLoader,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        steps: int,
        clip_ratio: float = 0.2,
        kl_penalty: float = 1.0,
        kl_cutoff: float | None = None,
        global_step: int = 0,
        scale_sigma: float = 0.1,
        device: torch.device | str = "cpu",
        use_amp: bool = True,
) -> int:
    scaler = torch.amp.GradScaler() if use_amp else None
    model.to(device)
    model.train()
    epochs = steps // len(loader)

    L_all = []
    prob_ratio_all = []
    kl_div_all = []
    entropy_all = []
    num_steps = 0

    model_state_dict = deepcopy(model.state_dict())
    optim_state_dict = deepcopy(optim.state_dict())

    start_time = time.time()
    for _ in range(epochs):
        stopping_early = False

        for data in loader:
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                optim.zero_grad()
                data.to(device)
                y_var = model(data)

            with torch.amp.autocast(device_type="cuda", enabled=False):
                y_var = y_var.float()
                var_params = data["var"].var_params.transpose(0, 1).float()
                log_prob_ref = data.log_prob.transpose(0, 1).float()
                y_var_ref = data["var"].y_var_ref.float()
                advantage = data.stats.transpose(0, 1).float()
                var_batch = data["lit"].batch[0::2]

                #log_prob_ref = policy.log_prob(y_var_ref, var_params, scale_sigma=scale_sigma)
                #log_prob = policy.log_prob(y_var, var_params, scale_sigma=scale_sigma)
                #L, prob_ratio = objective(log_prob, log_prob_ref, advantage[:, var_batch].unsqueeze(-1).tile(1, 1, 2), clip_ratio)

                log_prob = policy.log_prob(y_var, var_params, var_batch, scale_sigma=scale_sigma)
                L, prob_ratio = objective(log_prob, log_prob_ref, advantage, clip_ratio)

                kl_div = policy.kl_div(y_var, y_var_ref, var_batch, scale_sigma=scale_sigma)

                kl_div_mean = kl_div.mean()
                kl_div_all.append(kl_div_mean.item())
                L_total = L - kl_penalty * kl_div_mean

                last_kl_div = kl_div_mean.item()
                if kl_cutoff is not None and last_kl_div > kl_cutoff:
                    stopping_early = True
                    model.load_state_dict(model_state_dict)
                    optim.load_state_dict(optim_state_dict)
                    print(f"Early Stopping with KL div {last_kl_div:.4f}")
                    break
                else:
                    model_state_dict = deepcopy(model.state_dict())
                    optim_state_dict = deepcopy(optim.state_dict())

                entropy = policy.entropy(y_var, var_batch, scale_sigma=scale_sigma)
                entropy_all.append(entropy.item())

                if use_amp:
                    scaler.scale(L_total).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    L_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()

                L_all.append(L.item())
                prob_ratio_all.append(prob_ratio.detach().cpu().numpy())

                global_step += 1
                num_steps += 1

        if stopping_early:
            break

    metrics = {
        "train/L": np.mean(L_all),
        "train/prob_ratio": wandb.Histogram(np.concatenate(prob_ratio_all)),
        "train/lr": sched.get_last_lr()[0],
        "train/kl_div": np.mean(kl_div_all),
        "train/entropy": np.mean(entropy_all),
    }
    wandb.log(metrics, step=global_step)

    end_time = time.time()
    print(f"Optimized model for {num_steps} steps ({end_time - start_time:.2f} seconds)")

    sched.step()
    return global_step
