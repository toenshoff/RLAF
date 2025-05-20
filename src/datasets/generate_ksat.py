import math
import os

import numpy as np

from cnfgen import CNF
from cnfgen.families.randomformulas import RandomKCNF
from torch_geometric import seed_everything

from src.datasets.generate_data import CNFGenerator
from src.solving.solver import solve_cnf


class Balanced3SATGenerator(CNFGenerator):

    def __init__(
        self,
        num_var: int | tuple[int, int] = 50,
        target_num: int = 1000,
        data_dir: str = "data/3sat",
        num_workers: int = 8,
        epsilon: float = 0.2,
        seed: int = 1729,
    ):
        self.num_var = num_var if isinstance(num_var, tuple) else (num_var, num_var)
        self.epsilon = epsilon
        super(Balanced3SATGenerator, self).__init__(
            target_num=target_num,
            data_dir=data_dir,
            num_workers=num_workers,
            seed=seed
        )

    def _sample_formula(self):
        n = np.random.choice(range(self.num_var[0], self.num_var[1] + 1))
        alpha = (4.258 * n + 58.26 / (n**(2/3))) / n
        m_min = int((alpha - self.epsilon) * n)
        m_max = int((alpha + self.epsilon) * n)
        m = np.random.choice(range(m_min, m_max + 1))
        f = RandomKCNF(3, n, m)
        return f

    def _generate(self) -> tuple[CNF, CNF]:
        f = self._sample_formula()

        stats = solve_cnf(f.clauses(), solver="march")
        sat = stats["Result"] == "SATISFIABLE"
        if sat:
            f_pos = f
            f_neg = None
            target = False
        else:
            f_pos = None
            f_neg = f
            target = True

        while True:
            f = self._sample_formula()
            stats = solve_cnf(f.clauses(), solver="march")
            sat = stats["Result"] == "SATISFIABLE"
            if sat == target:
                break

        if target:
            f_pos = f
        else:
            f_neg = f

        return f_pos, f_neg


class Random3SATGenerator:

    def __init__(
        self,
        num_var: int | tuple[int, int] = 50,
        target_num: int = 1000,
        data_dir: str = "data/3sat",
        epsilon: float = 0.2,
        seed: int = 1729,
    ):
        self.num_var = num_var if isinstance(num_var, tuple) else (num_var, num_var)
        self.epsilon = epsilon
        self.target_num = target_num
        self.data_dir = data_dir
        self.seed = seed

    def _sample_formula(self):
        n = np.random.choice(range(self.num_var[0], self.num_var[1] + 1))
        alpha = (4.258 * n + 58.26 / (n**(2/3))) / n
        m_min = int((alpha - self.epsilon) * n)
        m_max = int((alpha + self.epsilon) * n)
        m = np.random.choice(range(m_min, m_max + 1))
        f = RandomKCNF(3, n, m)
        return f

    def generate_all(self):
        seed_everything(self.seed)
        os.makedirs(self.data_dir, exist_ok=True)

        for i in range(self.target_num):
            f = self._sample_formula()
            file_name = f'{self.data_dir}/3sat_{i}.cnf'
            with open(file_name, "w") as file:
                file.write(f.to_dimacs())
