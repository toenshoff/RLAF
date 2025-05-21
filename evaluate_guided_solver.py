import os

from omegaconf import DictConfig, OmegaConf
import hydra
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import DimacsCNFDataset
from src.model.model import load_checkpoint
from src.policy import policy
from src.policy.evaluate import sample_var_params, compute_solver_stats, var_params_from_target_prediction


def print_solver_metrics(solver_stats: pd.DataFrame) -> None:
    keys = ["decisions", "conflicts", "propagations", "restarts", "CPU time", "GPU time", "time"]
    metrics = {key: solver_stats[key].mean() for key in keys if key in solver_stats.columns}
    print(
        f"Metrics: \n"
        + "\n".join(f"{key}: {val:.2f}" for key, val in metrics.items())
    )


@hydra.main(version_base=None, config_path="configs", config_name="config_eval_guided_solver")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    if not cfg.is_supervised:
        model, transform, model_cfg = load_checkpoint(cfg.checkpoint, var_output=True)
    else:
        model, transform, model_cfg = load_checkpoint(cfg.checkpoint, var_output=False)

    dataset = DimacsCNFDataset(
        path=cfg.dataset.eval_path,
        transform=transform,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # warmup GPU for accurate time measurements
    warmup_steps = 4
    with torch.no_grad():
        for i, data in enumerate(loader):
            data.to(device)
            y_var = model(data)
            if not cfg.is_supervised:
                _ = policy.mode(y_var)
            if i >= warmup_steps:
                break

    if not cfg.is_supervised:
        data_list = sample_var_params(
            model=model,
            loader=loader,
            device=device,
            use_mode=True,
            num_samples=1,
            scale_sigma=model_cfg.scale_sigma,
            add_timing=True,
        )
    else:
        data_list = var_params_from_target_prediction(
            model=model,
            loader=loader,
            device=device,
            target=model_cfg.dataset.target,
            pred_scale=cfg.pred_scale,
            add_timing=True,
        )

    solver = model_cfg.solver.solver if cfg.solver.solver is None else cfg.solver.solver
    solver_params = cfg.solver.params
    print(solver_params)

    solver_stats = compute_solver_stats(
        dataset=dataset,
        data_list=data_list,
        num_workers=cfg.solver.num_workers,
        solver=solver,
        **solver_params,
    )

    gpu_time = pd.DataFrame({
        "cnf_id": [data.cnf_id.item() for data in data_list],
        "GPU time": [data.gpu_time for data in data_list]
    })
    solver_stats = solver_stats.merge(gpu_time, on="cnf_id")
    solver_stats["time"] = solver_stats["CPU time"] + solver_stats["GPU time"]

    print_solver_metrics(solver_stats)

    if cfg.save_file is not None:
        save_file = os.path.join(os.path.dirname(cfg.checkpoint), cfg.save_file)
        solver_stats.to_csv(save_file)


if __name__ == '__main__':
    main()
