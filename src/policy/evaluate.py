import time
from joblib import Parallel, delayed
from typing import Any, Literal


import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

from src.data.dataset import DimacsCNFDataset

from src.model.model import GNN
from src.solving.solver import solve_cnf
import src.policy.policy as policy


@torch.no_grad()
def sample_var_params(
    model: torch.nn.Module,
    loader: DataLoader,
    num_samples: int,
    max_num_batches: int = -1,
    device: torch.device | str = "cpu",
    use_mode: bool = False,
    scale_sigma: float = 0.1,
    add_timing: bool = False,
) -> list[HeteroData]:
    """
        Sample variable parameterizations with a given model and attach them to the PyG graphs.
        :param model: GNN model to use as a policy.
        :param loader: Dataloader with CNF formula graphs.
        :param num_samples: Number of parameterizations to sample for each graph.
        :param max_num_batches: If non-negative, sampling stops after this number of mini batches.
        :param device: Device to use for GNN inference.
        :param use_mode: If set to true, the mode of the policy will be used instead of random sampling.
        :param scale_sigma: Sigma parameters for the log-normal distributions of variable weights.
        :param add_timing: If true, the GNN forward pass will be timed.
        :return: list of HeteroData objects where each is an individual CNF graph with attached variables parameterization, and corresponding log probs.
    """
    model.to(device)
    model.eval()

    data_list_all = []
    for i, data in enumerate(loader):
        if add_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        data.to(device)
        lit_batch = data["lit"].batch
        var_batch = lit_batch[0::2]

        y_var = model(data)

        if use_mode:
            var_params = policy.mode(y_var=y_var, scale_sigma=scale_sigma)
        else:
            var_params = policy.sample(y_var=y_var, num_samples=num_samples, scale_sigma=scale_sigma)

        if add_timing:
            end.record()
            torch.cuda.synchronize()
            gpu_time = start.elapsed_time(end) / 1.0e3

        log_prob = policy.log_prob(y_var, var_params, var_batch, scale_sigma=scale_sigma)

        data.to("cpu")
        data_list = data.to_data_list()

        y_var = y_var.to("cpu")
        log_prob = log_prob.to("cpu").transpose(0, 1)
        var_batch = var_batch.to("cpu")
        var_params = var_params.to("cpu").transpose(0, 1)

        y_var = unbatch(y_var, var_batch)
        var_params = unbatch(var_params, var_batch)

        for j, data in enumerate(data_list):
            data.log_prob = log_prob[j]
            data["var"].y_var_ref = y_var[j]
            data["var"].num_nodes = data["lit"].num_nodes // 2
            data["var"].var_params = var_params[j]

            if add_timing:
                data.gpu_time = gpu_time

            data_list_all.append(data)

        if max_num_batches > -1 and i + 1 >= max_num_batches:
                break

    return data_list_all


@torch.no_grad()
def var_params_from_target_prediction(
    model: GNN,
    loader: DataLoader,
    target: Literal["backbone", "core"],
    device: torch.device | str = "cpu",
    pred_scale: float = 1.0,
    add_timing: bool = False,
) -> list[HeteroData]:
    """
        Map supervised literal predictions to variable parameterizations as described in the paper.
        :param model: GNN model to use as a policy.
        :param loader: Dataloader with CNF formula graphs.
        :param target: Supervised prediction target (backbone or core).
        :param device: Device to use for GNN inference.
        :param add_timing: If true, the GNN forward pass will be timed.
        :param pred_scale: Scaling factor for the variable weights.
        :return: list of HeteroData objects where each is an individual CNF graph with attached variables parameterization.
    """

    model.to(device)
    model.eval()

    data_list_all = []
    for i, data in enumerate(loader):
        if add_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        data.to(device)
        lit_batch = data["lit"].batch
        var_batch = lit_batch[0::2]

        y_lit = model(data).flatten()
        y_lit = torch.sigmoid(y_lit)
        y_lit_neg = y_lit[0::2]
        y_lit_pos = y_lit[1::2]
        if target == "backbone":
            phase = (y_lit_pos > y_lit_neg).float()
            weight = pred_scale * 0.5 * (y_lit_pos + y_lit_neg)
        elif target == "core":
            weight = pred_scale * 0.5 * (y_lit_pos + y_lit_neg)
            phase = torch.ones_like(weight)
        else:
            raise ValueError(f"Unknown target label {target}")
        var_params = torch.stack([phase, weight], dim=1).unsqueeze(0)

        if add_timing:
            end.record()
            torch.cuda.synchronize()
            gpu_time = start.elapsed_time(end) / 1.0e3

        data.to("cpu")
        data_list = data.to_data_list()

        var_batch = var_batch.to("cpu")
        var_params = var_params.to("cpu").transpose(0, 1)
        var_params = unbatch(var_params, var_batch)

        for j, data in enumerate(data_list):
            data["var"].num_nodes = data["lit"].num_nodes // 2
            data["var"].var_params = var_params[j]

            if add_timing:
                data.gpu_time = gpu_time

            data_list_all.append(data)

    return data_list_all


def solver_pool_fn(args: tuple) -> dict:
    cnf_idx, assign_idx, cnf, var_params, solver_params = args
    stats_dict = solve_cnf(cnf.clauses, var_params, **solver_params)
    stats_dict["cnf_id"] = cnf_idx
    stats_dict["sample_id"] = assign_idx
    return stats_dict


def compute_solver_stats(
        dataset: DimacsCNFDataset,
        data_list: list[HeteroData],
        num_workers: int = 8,
        **solver_params: Any,
) -> pd.DataFrame:
    """
    Run sat solver and collect runtime statistics for a set of CNFs.
    :param dataset: Underlying CNF dataset.
    :param data_list: List of HeteroData objects containing variable parameterizations.
    :param num_workers: Number of CPU cores to use for solving.
    :param solver_params: Additional solver parameters (only used for glucose).
    :return: A pandas DataFrame containing solver run statistics.
    """

    def iter_inputs():
        for data in data_list:
            cnf_id = data.cnf_id.item()
            var_params_all = data["var"].var_params.numpy()
            for j in range(var_params_all.shape[1]):
                cnf = dataset.cnf_list[cnf_id]
                var_params = var_params_all[:, j]
                args = (cnf_id, j, cnf, var_params, solver_params)
                yield args

    total_len = sum(data["var"].var_params.shape[1] for data in data_list)

    start_time = time.time()

    stats_dicts = Parallel(n_jobs=num_workers)(
        delayed(solver_pool_fn)(inp)
        for inp in iter_inputs()
    )

    end_time = time.time()
    print(f"Solved {total_len} formulas in {end_time - start_time:.2f} seconds")

    solver_stats = pd.DataFrame.from_records(stats_dicts)
    solver_stats["file"] = [dataset.id_to_file[i] for i in solver_stats["cnf_id"]]

    return solver_stats
