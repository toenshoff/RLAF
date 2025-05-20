import os

import networkx as nx
import numpy as np
from scipy.special import binom

from cnfgen import CNF
from cnfgen.families.coloring import GraphColoringFormula

from pysat.formula import CNF as PSCNF
from pysat.solvers import Glucose42
from torch_geometric import seed_everything

from src.datasets.generate_balanced import BalancedGenerator


class BalancedColoringGenerator(BalancedGenerator):

    def __init__(
        self,
        num_nodes: int | tuple[int, int] = 50,
        target_num: int = 1000,
        data_dir: str = "data/coloring",
        num_workers: int = 8,
        seed: int = 0,
    ):
        self.num_nodes = num_nodes if isinstance(num_nodes, tuple) else (num_nodes, num_nodes)
        super(BalancedColoringGenerator, self).__init__(target_num=target_num, data_dir=data_dir, num_workers=num_workers, seed=seed)

    def _sample_formula(self):
        n = np.random.choice(range(self.num_nodes[0], self.num_nodes[1] + 1))
        m = 4.67 * n / 2.0 # https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.031131
        p = m / binom(n, 2)
        G = nx.gnp_random_graph(n, p)
        f = GraphColoringFormula(G, 3)
        return f

    def _generate(self) -> tuple[CNF, CNF]:
        f = self._sample_formula()

        with Glucose42(bootstrap_with=PSCNF(from_clauses=f.clauses())) as solver:
            if solver.solve():
                f_pos = f
                f_neg = None
                target = False
            else:
                f_pos = None
                f_neg = f
                target = True

            while True:
                f = self._sample_formula()
                with Glucose42(bootstrap_with=PSCNF(from_clauses=f.clauses())) as solver:
                    if solver.solve() == target:
                        break
            if target:
                f_pos = f
            else:
                f_neg = f

            return f_pos, f_neg


class Random3ColGenerator:

    def __init__(
        self,
        num_nodes: int | tuple[int, int] = 50,
        target_num: int = 1000,
        data_dir: str = "data/coloring",
        seed: int = 1729,
    ):
        self.num_nodes = num_nodes if isinstance(num_nodes, tuple) else (num_nodes, num_nodes)
        self.target_num = target_num
        self.data_dir = data_dir
        self.seed = seed

    def _sample_formula(self):
        n = np.random.choice(range(self.num_nodes[0], self.num_nodes[1] + 1))
        m = 4.67 * n / 2.0 # https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.031131
        p = m / binom(n, 2)
        G = nx.gnp_random_graph(n, p)
        f = GraphColoringFormula(G, 3)
        return f

    def generate_all(self):
        seed_everything(self.seed)
        os.makedirs(self.data_dir, exist_ok=True)

        for i in range(self.target_num):
            f = self._sample_formula()
            file_name = f'{self.data_dir}/3col_{i}.cnf'
            with open(file_name, "w") as file:
                file.write(f.to_dimacs())
