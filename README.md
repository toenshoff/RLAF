# RLAF

## Install
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

Build solvers from source:
```bash
bash build_sovlers.sh
```
