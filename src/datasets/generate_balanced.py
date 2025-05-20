import os
from multiprocessing import Pool

from cnfgen import CNF
from torch_geometric import seed_everything
from tqdm import tqdm


class CNFGenerator:

    def __init__(
        self,
        target_num: int = 1000,
        data_dir: str = "data/subgraph",
        num_workers: int = 8,
        seed: int = 1729,
    ):
        self.target_num = target_num
        self.data_dir = data_dir
        self.num_workers = num_workers

        self.num_sat = 0
        self.num_unsat = 0

        self.seed = seed

    def _generate(self) -> tuple[CNF | None, CNF | None]:
       raise NotImplementedError

    def generate_and_write(self, idx: int) -> None:
        f_pos, f_neg = self._generate()
        if f_pos is not None and self.num_sat < self.target_num:
            file_name = f'{self.data_dir}/sat/sat_{idx}.cnf'
            with open(file_name, "w") as f:
                f.write(f_pos.to_dimacs())
            self.num_sat += 1
        if f_neg is not None and self.num_unsat < self.target_num:
            file_name = f'{self.data_dir}/unsat/unsat_{idx}.cnf'
            with open(file_name, "w") as f:
                f.write(f_neg.to_dimacs())
            self.num_unsat += 1

    def generate_all(self) -> None:
        seed_everything(self.seed)

        os.makedirs(os.path.join(self.data_dir, "sat"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "unsat"), exist_ok=True)

        if self.num_workers == 0:
            for i in tqdm(range(self.target_num), total=self.target_num, desc="Generating CNF formulas"):
                self.generate_and_write(i)
        else:
            with Pool(self.num_workers) as p:
                _ = list(tqdm(
                    p.imap_unordered(self.generate_and_write, range(self.target_num)),
                    total=self.target_num,
                    desc="Generating CNF formulas"
                ))
