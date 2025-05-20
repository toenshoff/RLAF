from copy import copy
from multiprocessing import Pool
from typing import Literal

import numpy as np
import pandas as pd

from typing import Callable, Optional
from glob import glob
from tqdm import tqdm
import torch
from torch_geometric.data import HeteroData, Dataset
from pysat.formula import CNF

from src.data.cnf import cnf_to_pyg
from src.solving.backbone import get_backbone_lits
from src.solving.core import get_core_vars


class DimacsCNFDataset(Dataset):
    """
    This dataset class provides the functionality for:
    1. Loading a collection of DIMACS CNF formulas from a given path or glob pattern (e.g., "*.cnf").
    2. Converting each loaded CNF to a PyG graph using `cnf_to_pyg`.
    3. Applying an optional transform function to each PyG graph.
    """

    def __init__(
            self,
            path: str,
            transform: Optional[Callable] = None,
    ):
        """
        :param path: A glob pattern or directory path pointing to DIMACS CNF files.
        :param transform: An optional PyG transform to apply to each graph.
        :param binary_cls_as_edges: Whether to interpret 2-literal clauses as edges in the final graph.
        """
        super().__init__()
        self.path = path
        self.transform = transform

        # 1) Load CNFs from the specified path
        self.cnf_dict = self._load_dimacs_files(self.path)

        # 2) Convert CNFs to PyG data objects
        self.id_to_file = {i: fn for i, fn in enumerate(self.cnf_dict.keys())}
        self.cnf_list = [self.cnf_dict[self.id_to_file[i]] for i in range(len(self.cnf_dict))]
        self.data_list = self._convert_to_pyg()

    def _load_dimacs_files(self, path: str) -> dict[str, CNF]:
        """ Loads .dimacs files from the given path (glob pattern). """
        files = list(glob(path))
        files.sort()
        if not files:
            raise ValueError(f"No DIMACS files found for path/pattern: {path}")

        cnf_dict = {}
        for f in tqdm(files, desc=f"Loading DIMACS files from '{path}'"):
            cnf_dict[f] = CNF(from_file=f)
        return cnf_dict

    def _convert_to_pyg(self) -> list[HeteroData]:
        """ Converts each CNF in self.cnf_list into a PyG data object via cnf_to_pyg. """
        enumerated_cnfs = list(enumerate(self.cnf_list))

        data_list = []
        for i, cnf in tqdm(enumerated_cnfs, desc="Converting CNFs to PyG"):
            data = cnf_to_pyg(f=cnf.clauses, num_var=cnf.nv)
            # Optionally apply a transform
            if self.transform is not None:
                data = self.transform(data)
            # Store the index as a tensor
            data.cnf_id = torch.tensor(i, dtype=torch.long)
            data_list.append(data)
        return data_list

    def __getitem__(self, idx: int) -> HeteroData:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)

    def len(self) -> int:
        return self.__len__()

    def get(self, idx: int) -> HeteroData:
        return self.__getitem__(idx)


def _label_fn(args: tuple[str, str]) -> tuple[str, np.ndarray]:
    file, target = args
    if target == "backbone":
        lit_mask = get_backbone_lits(file, as_array=True, cache=True)
    elif target == "core":
        var_mask = get_core_vars(file, cache=True, keep_aux_files=False)
        lit_mask = np.repeat(var_mask, 2)
    else:
        raise ValueError(f"Unknown target label {target}")
    return file, lit_mask


class LabeledDataset(DimacsCNFDataset):
    """ Dataset for supervised training """

    def __init__(
            self,
            path: str,
            transform: Optional[Callable] = None,
            target: str = "backbone",
            num_workers: int = 0,
    ):
        super(LabeledDataset, self).__init__(path, transform)
        self.target = target
        self.num_workers = num_workers

        inputs = [(file, self.target) for file in self.id_to_file.values()]
        label_map = {}
        if self.num_workers <= 1:
            for args in tqdm(inputs, desc=f"Adding {target} target to data"):
                file, target = _label_fn(args)
                label_map[file] = target
        else:
            # Run in parallel via multiprocessing
            with Pool(self.num_workers) as pool:
                # imap is a lazy iterator, so we can wrap it in tqdm for progress
                results_iter = pool.imap(_label_fn, inputs)
                for file, target in tqdm(results_iter, total=len(inputs), desc=f"Adding {target} target to data"):
                    label_map[file] = target

        for i, data in enumerate(self.data_list):
            file = self.id_to_file[i]
            target = label_map[file]
            data["lit"].target = torch.tensor(target, dtype=torch.float32)


class PreferenceTrainingDataset(Dataset):

    def __init__(
            self,
            solver_stats: pd.DataFrame,
            data_list: list[HeteroData],
            target_stat: str = "decisions",
            objective: Literal["minimize", "maximize"] = "minimize",
    ):
        self.solver_stats = solver_stats.sort_values(by=["cnf_id", "sample_id"], ascending=[True, True])

        self.data = []
        for data in data_list:
            data = copy(data)
            cnf_id = data.cnf_id.item()

            cnf_stats = self.solver_stats[self.solver_stats["cnf_id"] == cnf_id]
            stats = torch.tensor(cnf_stats[target_stat].to_numpy())

            if objective == "minimize":
                idx = torch.argsort(-stats)
            else:
                idx = torch.argsort(stats)

            var_params = data["var"].var_params
            log_prob = data.log_prob

            data["var"].var_params = var_params[:, idx]
            data.log_prob = log_prob[idx].unsqueeze(0)
            data.stats = stats[idx].unsqueeze(0)

            self.data.append(data)

        super(PreferenceTrainingDataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def len(self) -> int:
        return self.__len__()

    def get(self, idx: int) -> HeteroData:
        return self.__getitem__(idx)
