## RLAF

## Setup
We recommend to use the package manager [`conda`](https://docs.conda.io/en/latest/). Once installed run
```bash
conda create -n rlaf python=3.12
conda activate rlaf
```

Install all dependencies via
```bash
pip install -e .
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```
Replace `cu124` with your cuda version or `cpu` when working without an nvidia gpu.

Build sat solvers from source:
```bash
bash build_sovlers.sh
```

Unzip datasets
```bash
unzip data.zip
```

## Training

To train our main RLAF-guided glucose models run the following
```bash
python train_rlaf.py model_name=GNN_Glucose_3SAT solver.solver=glucose dataset.train_path=data/training/3sat/*/*.cnf dataset.val_path=data/validation/3sat/*/*.cnf optim.lr=0.0001 training.kl_penalty=0.1
python train_rlaf.py model_name=GNN_Glucose_Coloring solver.solver=glucose dataset.train_path=data/training/coloring/*/*.cnf dataset.val_path=data/validation/coloring/*/*.cnf optim.lr=0.00005 training.kl_penalty=1.1
python train_rlaf.py model_name=GNN_Glucose_Crypto solver.solver=glucose dataset.train_path=data/training/crypto/*.cnf dataset.val_path=data/validation/crypto/*.cnf optim.lr=0.00005 training.kl_penalty=0.1
```

To train our main RLAF-guided march models run the following
```bash
python train_rlaf.py model_name=GNN_March_3SAT solver.solver=march dataset.train_path=data/training/3sat/*/*.cnf dataset.val_path=data/validation/3sat/*/*.cnf optim.lr=0.0001 training.kl_penalty=1.0
python train_rlaf.py model_name=GNN_March_Coloring solver.solver=march dataset.train_path=data/training/coloring/*/*.cnf dataset.val_path=data/validation/coloring/*/*.cnf optim.lr=0.00001 training.kl_penalty=0.1
python train_rlaf.py model_name=GNN_March_Crypto solver.solver=march dataset.train_path=data/training/crypto/*.cnf dataset.val_path=data/validation/crypto/*.cnf optim.lr=0.00001 training.kl_penalty=0.1
```

Training the supervised models:
```bash
python train_supervised.py model_name=GNN_Backbone_3SAT target=backbone dataset.train_path=data/training/3sat/sat/*.cnf dataset.val_path=data/validation/3sat/sat/*.cnf
python train_supervised.py model_name=GNN_Core_Coloring target=core dataset.train_path=data/training/coloring/*/*.cnf dataset.val_path=data/validation/coloring/unsat/*.cnf
python train_supervised.py model_name=GNN_Core_Crypto target=core dataset.train_path=data/training/crypto/*.cnf dataset.val_path=data/validation/crypto/*.cnf
```

## Evaluation

Run the following script to evaluate the RLAF-guided Glucose solver on 3SAT problems with 450 variables:
```bash
python evaluate_guided_solver.py model_name=GNN_Glucose_3SAT dataset.eval_path=data/test/3sat/450/*.cnf
```

To evaluate the supervised models use the `ìs_supervised` flag:
```bash
python evaluate_guided_solver.py model_name=GNN_Backbone_3SAT dataset.eval_path=data/test/3sat/450/*.cnf is_supervised=True pred_scale=10.0
```

Finally, to evaluate the base solver run the follwing script:
```bash
python evaluate_base_solver.py solver.solver=glucose dataset.eval_path=data/test/3sat/450/*.cnf
```
