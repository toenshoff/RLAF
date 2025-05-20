import os
from glob import glob
from typing import Any
from joblib import Parallel, delayed

import hydra
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

from src.data.cnf import cnf_to_adj
from src.data.io_utils import load_dimacs_cnf
from src.solving.solver import solve_cnf
from src.solving.core import get_core_vars
from src.solving.backbone import get_backbone_lits


def unsat_core_weights(f_path: str, core_weight: float = 1.0, non_core_weight: float = 0.1) -> None | np.ndarray:
    core_vars = get_core_vars(f_path)
    if core_vars is None:
        return None
    core_mask = core_vars > 0.0
    var_params = np.ones((core_vars.shape[0], 3))
    var_params[core_mask, 1:] = core_weight
    var_params[~core_mask, 1:] = non_core_weight
    return var_params


def backbone_weights(f_path: str, backbone_weight: float = 1.0, non_backbone_weight: float = 0.01) -> None | np.ndarray:
    backbone_lits = get_backbone_lits(f_path, as_array=True, cache=True).astype(np.bool_)
    backbone_var = backbone_lits[0::2] | backbone_lits[1::2]
    var_params = np.ones((backbone_var.shape[0], 3))
    var_params[backbone_var, 1:] = backbone_weight
    var_params[~backbone_var, 1:] = non_backbone_weight
    var_params[backbone_lits[0::2], 0] = 0
    var_params[backbone_lits[1::2], 0] = 1
    return var_params


def centrality_weights(f):
    adj = cnf_to_adj(f).to_dense().numpy()
    adj = adj[:, 0::2] + adj[:, 1::2]
    A = adj.transpose() @ adj
    A[A > 0] = 1.0
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    x = np.array(list(nx.centrality.eigenvector_centrality(G).values()))
    var_params = np.ones((A.shape[0], 3))
    var_params[:, 2] = x
    return var_params


def solver_pool_fn(args: tuple) -> dict:
    cnf, assignment, solver_params, f_path = args
    stats_dict = solve_cnf(cnf, assignment, **solver_params)
    stats_dict["file"] = f_path
    return stats_dict


def compute_solver_stats(
        cnf_list: list[list[list[int]]],
        files: list[str],
        num_workers: int = 8,
        add_var_params: str = "none",
        **solver_params: Any,
) -> pd.DataFrame:

    def iter_inputs():
        for i, f in enumerate(cnf_list):
            f_path = files[i]
            if add_var_params == "centrality":
                var_params = centrality_weights(f)
            elif add_var_params == "core":
                var_params = unsat_core_weights(f_path)
            elif add_var_params == "backbone":
                var_params = backbone_weights(f_path)
            elif add_var_params == "none":
                var_params = None
            else:
                raise ValueError(f"Unknown type of variable parameters {add_var_params}")
            args = (f, var_params, solver_params, f_path)
            yield args

    stats_dicts = Parallel(n_jobs=num_workers, verbose=5)(
        delayed(solver_pool_fn)(inp)
        for inp in iter_inputs()
    )

    solver_stats = pd.DataFrame.from_records(stats_dicts)
    return solver_stats


def print_solver_metrics(solver_stats: pd.DataFrame) -> None:
    keys = ["decisions", "conflicts", "propagations", "restarts", "CPU time"]
    metrics = {key: solver_stats[key].mean() for key in keys if key in solver_stats.columns}

    print(
        f"Metrics: \n"
        + "\n".join(f"{key}: {val:.2f}" for key, val in metrics.items())
    )


@hydra.main(version_base=None, config_path="configs", config_name="config_eval_solver")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    path = cfg.dataset.eval_path
    files = list(glob(path))
    cnf_list = [load_dimacs_cnf(f) for f in tqdm(files, desc=f"Loading DIMACS files from {path}")]

    solver_stats = compute_solver_stats(
        cnf_list=cnf_list,
        files=files,
        num_workers=cfg.solver.num_workers,
        add_var_params=cfg.solver.add_var_params,
        solver=cfg.solver.solver,
        **cfg.solver.params,
    )
    solver_stats["time"] = solver_stats["CPU time"]

    print_solver_metrics(solver_stats)

    os.makedirs(cfg.save_dir, exist_ok=True)
    if cfg.save_file is not None:
        save_file = os.path.join(cfg.save_dir, cfg.save_file)
        solver_stats.to_csv(save_file)


if __name__ == '__main__':
    main()
