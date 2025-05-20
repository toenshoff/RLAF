from src.datasets.generate_col import BalancedColoringGenerator, Random3ColGenerator
from src.datasets.generate_ksat import Balanced3SATGenerator, Random3SATGenerator


if __name__ == "__main__":
    Balanced3SATGenerator(data_dir="data/training/3sat", target_num=10000, num_var=(200, 200), num_workers=10, seed=0).generate_all()
    Balanced3SATGenerator(data_dir="data/validation/3sat", target_num=100, num_var=(200, 200), num_workers=10, seed=1).generate_all()
    for n in [350, 300, 450]:
        Random3SATGenerator(data_dir=f"data/test/3sat/{n}", target_num=200, num_var=(n, n), seed=2).generate_all()

    BalancedColoringGenerator(data_dir="data/training/coloring", target_num=10000, num_nodes=(300, 300), num_workers=10, seed=0).generate_all()
    BalancedColoringGenerator(data_dir="data/validation/coloring", target_num=100, num_nodes=(300, 300), num_workers=10, seed=1).generate_all()
    for n in [400, 500, 600]:
        Random3ColGenerator(data_dir=f"data/test/coloring/{n}", target_num=200, num_nodes=(n, n), seed=2).generate_all()

    # for the generation of cryptographic instances we refer to the shell script src/datasets/generate_crypto.sh
