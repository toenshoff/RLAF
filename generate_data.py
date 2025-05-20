from src.datasets.generate_sr import SRSATGenerator
from src.datasets.generate_sgen1 import Sgen1Generator
from src.datasets.generate_clique import CliqueGenerator
from src.datasets.generate_domset import DomSetGenerator
from src.datasets.generate_col import BalancedColoringGenerator, Random3ColGenerator
from src.datasets.generate_ksat import Balanced3SATGenerator, Random3SATGenerator
from src.datasets.generate_pitfall import PitfallGenerator
from src.datasets.generate_subformula import SubformulaGenerator


if __name__ == "__main__":
    #KSATGenerator(data_dir="data/validation/3SAT", k=3, target_num=100, num_var=200, num_workers=8).generate_all()
    #KSATGenerator(data_dir="data/validation/4SAT", k=4, target_num=100, num_var=75, num_workers=8).generate_all()
    #KSATGenerator(data_dir="data/validation/5SAT", k=5, target_num=100, num_var=50, num_workers=8).generate_all()
    #SRSATGenerator(data_dir="data/validation/SR", target_num=100, num_var=200, num_workers=8).generate_all()

    #KSATGenerator(data_dir="data/training/3SAT", k=3, target_num=5000, num_var=200, num_workers=8).generate_all()
    #KSATGenerator(data_dir="data/training/4SAT", k=4, target_num=5000, num_var=75, num_workers=8).generate_all()
    #KSATGenerator(data_dir="data/training/5SAT", k=5, target_num=5000, num_var=50, num_workers=8).generate_all()

    #Sgen1Generator(data_dir="data/training/SGen100", num_var_sat=100, num_var_unsat=-1, target_num=10000, num_workers=4, seed=0).generate_all()
    #Sgen1Generator(data_dir="data/validation/SGen100", num_var_sat=100, num_var_unsat=-1, target_num=100, num_workers=4, seed=1).generate_all()
    # Sgen1Generator(data_dir="data/validation/SGen1", target_num=100, num_workers=1).generate_all()
    #Sgen1Generator(data_dir="data/test/SGen1_70", target_num=100, num_workers=8, num_var_unsat=70, num_var_sat=60).generate_all()

    #SRSATGenerator(data_dir="data/training/SRNew", target_num=5000, num_var=(100, 200), num_workers=8).generate_all()
    #SRSATGenerator(data_dir="data/validation/SRNew", target_num=100, num_var=(100, 200), num_workers=8).generate_all()

    #KSATGenerator(data_dir="../../data/test/3SAT", k=3, target_num=100, num_var=300, num_workers=8).generate_all()
    #SRSATGenerator(data_dir="data/training/SROG", target_num=5000, num_var=500, num_workers=30).generate_all()
    #SRSATGenerator(data_dir="data/validation/SROG", target_num=100, num_var=400, num_workers=1).generate_all()

    #CliqueGenerator(data_dir="data/training/clique", target_num=5000, num_nodes=(20, 50), clique_size=(5, 10), num_workers=8).generate_all()
    #CliqueGenerator(data_dir="data/validation/clique", target_num=100, num_nodes=(20, 50), clique_size=(5, 10), num_workers=8).generate_all()

    #BalancedColoringGenerator(data_dir="data/training/coloring", target_num=10000, num_nodes=(300, 300), num_workers=15, seed=0).generate_all()
    #BalancedColoringGenerator(data_dir="data/validation/coloring", target_num=100, num_nodes=(300, 300), num_workers=15, seed=1).generate_all()
    #Random3ColGenerator(data_dir="data/test/coloring/400", target_num=200, num_nodes=(400, 400), seed=2).generate_all()
    #Random3ColGenerator(data_dir="data/test/coloring/500", target_num=200, num_nodes=(500, 500), seed=2).generate_all()
    #Random3ColGenerator(data_dir="data/test/coloring/550", target_num=200, num_nodes=(550, 550), seed=2).generate_all()

    #Balanced3SATGenerator(data_dir="data/validation/3sat/200", target_num=100, num_var=(200, 200), epsilon=0.0, num_workers=10, seed=1).generate_all()
    #Balanced3SATGenerator(data_dir="data/validation/3sat/250", target_num=100, num_var=(250, 250), epsilon=0.0, num_workers=10, seed=1).generate_all()
    #Balanced3SATGenerator(data_dir="data/training/3sat/200", target_num=10000, num_var=(200, 200), epsilon=0.0, num_workers=10, seed=0).generate_all()
    #Balanced3SATGenerator(data_dir="data/training/3sat/250", target_num=10000, num_var=(250, 250), epsilon=0.0, num_workers=10, seed=0).generate_all()
    for n in [450, 500]:
        Random3SATGenerator(data_dir=f"data/test/3sat/{n}", target_num=200, num_var=(n, n), epsilon=0.0, seed=2).generate_all()

    #KCNFGenerator(data_dir="data/training/3sat250", target_num=10000, num_var=(250, 250), num_workers=8, seed=1).generate_all()
    #KCNFGenerator(data_dir="data/validation/3sat250", target_num=100, num_var=(250, 250), num_workers=8, seed=0).generate_all()
    #KCNFGenerator(data_dir="data/validation/3sat", target_num=100, num_var=(200, 200), num_workers=0, seed=1).generate_all()
    #KCNFGenerator(data_dir="data/training/3sat", target_num=10000, num_var=(200, 200), num_workers=0, seed=0).generate_all()
