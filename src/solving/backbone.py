import glob
import os.path
from multiprocessing import Pool

import numpy as np
from pysat.solvers import Glucose42
from pysat.formula import CNF
from tqdm import tqdm

from src.data.cnf import to_normalized_lit


def get_backbone_lits(file: str, as_array: bool = False, cache: bool = False) -> list[int] | np.ndarray:
    if cache:
        cache_file = f"{file}.backbone.npy"
        if os.path.exists(cache_file):
            return np.load(cache_file)
    else:
        cache_file = None

    cnf = CNF(from_file=file)
    backbone_lits = []
    with Glucose42(bootstrap_with=cnf) as solver:
        if solver.solve():
            backbone_lits = []
            for x in range(1, cnf.nv + 1):
                for l in (x, -x):
                    if not solver.solve(assumptions=[-l]):
                        backbone_lits.append(l)
                        break
    if not as_array:
        return backbone_lits
    else:
        backbone_mask = np.zeros((2 * cnf.nv,))
        if len(backbone_lits) > 0:
            backbone_lits_idx = np.array([to_normalized_lit(l) for l in backbone_lits])
            backbone_mask[backbone_lits_idx] = 1.0
        backbone_lits = backbone_mask

    if cache:
        np.save(cache_file, backbone_lits)
    return backbone_lits


def save_backbone(file: str, save_dir: str):
    backbone_lits = get_backbone_lits(file)
    fname = os.path.basename(file)
    np.save(os.path.join(save_dir, f"{fname}.backbone.npy"), backbone_lits)


def precompute_all_backbones(dimacs_dir: str, num_workers: int = 8):
    file_list = glob.glob(os.path.join(dimacs_dir, "*.cnf"))

    backbone_dir = os.path.join(dimacs_dir, "backbone")
    os.makedirs(backbone_dir, exist_ok=True)

    if num_workers == 0:
        for file in tqdm(file_list, desc="Computing Backbones"):
            save_backbone(file, backbone_dir)
    else:
        inputs = [(file, backbone_dir) for file in file_list]

        with Pool(num_workers) as p:
            _ = list(tqdm(
                p.imap_unordered(save_backbone, inputs),
                total=len(file_list),
                desc="Computing Backbones"
            ))


if __name__ == '__main__':
    lits = get_backbone_lits("../../data/validation/random_3sat/sat/sat_4.cnf", as_array=True)
    print(lits)
