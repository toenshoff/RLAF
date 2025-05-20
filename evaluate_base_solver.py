import os
from glob import glob
from typing import Any

from joblib import Parallel, delayed
import hydra
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.data.io_utils import load_dimacs_cnf
from src.solving.solver import solve_cnf



def solver_pool_fn(args: tuple) -> dict:
    cnf, assignment, solver_params, f_path = args
    stats_dict = solve_cnf(cnf, assignment, **solver_params)
    stats_dict["file"] = f_path
    return stats_dict


def compute_solver_stats(
        cnf_list: list[list[list[int]]],
        files: list[str],
        num_workers: int = 8,
        **solver_params: Any,
) -> pd.DataFrame:

    def iter_inputs():
        for i, f in enumerate(cnf_list):
            f_path = files[i]
            var_params = None
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


@hydra.main(version_base=None, config_path="configs", config_name="config_eval_base_solver")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    path = cfg.dataset.eval_path
    files = list(glob(path))
    cnf_list = [load_dimacs_cnf(f) for f in tqdm(files, desc=f"Loading DIMACS files from {path}")]

    solver_stats = compute_solver_stats(
        cnf_list=cnf_list,
        files=files,
        num_workers=cfg.solver.num_workers,
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
