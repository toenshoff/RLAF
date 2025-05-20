import os
from typing import Any

import numpy as np
import subprocess
import tempfile

from src.data.io_utils import load_dimacs_cnf


SOLVER_BIN_PATHS = {
    "glucose": "solvers/glucose/simp/glucose_static",
    "glucose_weighted": "solvers/glucose_weighted/simp/glucose_static",
    "march": "solvers/march/march_nh",
    "march_weighted": "solvers/march_weighted/march_nh",
}


STATS = ["decisions", "conflicts", "propagations", "restarts", "CPU time"]


def stdout_to_results_dict(stdout: str) -> dict[str, Any]:
    """ Parse a string of solver outputs into a results dictionary """
    stats = {}

    for line in stdout.splitlines():
        line = line.strip()
        for stat in STATS:
            if line.startswith(f"c {stat}") or line.startswith(stat):
                _, value_part = line.split(':', 1)
                value = float(value_part.strip().split()[0])
                stats[stat] = value
            elif line.startswith("s"):
                stats["Result"] = line.split()[1]

    return stats


def cnf_to_dimacs(f: list[list[int]], var_params: np.ndarray | None = None) -> str:
    """
    :param f: CNF formula formatted as a list of lists of signed integers. Each integer is one literal.
    :param var_params: Variable parameterization W. This is an array of shape [num_vars, 2], where
    var_params[i,0] is the polarity and var_params[i,1] is the variable weight.
    :return: The DIMACS string that represents the CNF formula. If `var_params` is `None`, then the
     polarities and weights are included in the string as a comment line starting with 'c weight'.
     Our extended solvers are built to parse polarities and weights from this line, if provided.
    """
    # Determine the maximum variable index
    variables = set(abs(lit) for clause in f for lit in clause)
    num_variables = max(variables)
    num_clauses = len(f)

    lines = [f"p cnf {num_variables} {num_clauses}"]
    if var_params is not None:
        assert num_variables == var_params.shape[0], f"{num_variables} != {var_params.shape}[0]"

        params = ["c weight"]
        for i in range(num_variables):
            weight = float(var_params[i, 1])
            sgn = 1 if var_params[i, 0] > 0 else -1
            params.append(f"{sgn * weight:.4f}")
        lines.append(" ".join(params))

    for clause in f:
        clause_line = " ".join(map(str, clause)) + " 0"
        lines.append(clause_line)
    return "\n".join(lines)


def solve_cnf(
        f: list[list[int]],
        var_params: np.ndarray | None = None,
        seed: int | None = 1,
        solver: str = "glucose",
        **params: Any,
) -> dict[str, Any]:
    """
    Solves a given CNF formula with a specified solver. With weights and polarities are provided then the guided version of the solver is used.
    :param f: CNF formula formatted as a list of lists of signed integers. Each integer is one literal.
    :param var_params: Variable parameterization W. This is an array of shape [num_vars, 2], where
    var_params[i,0] is the polarity and var_params[i,1] is the variable weight.
    :param seed: Seed for random number generation passed to the glucose solver.
    :param solver: Solver to use. Either glucose or march,
    :param params: Solver CLI parameters. Only used with glucose.
    :return: A dictionary that with solver statistics, including the results, the runtime and the number of decisions required.
    """

    dimacs_str = cnf_to_dimacs(f, var_params=var_params)

    if solver != "march":
        if var_params is None:
            bin_path = SOLVER_BIN_PATHS["glucose"]
        else:
            bin_path = SOLVER_BIN_PATHS["glucose_weighted"]

        call = [f"{bin_path}"]

        if seed is not None:
            assert seed > 0
            call.append(f"-rnd-seed={seed}")

        call += [f"-{key}={val}" for key, val in params.items()]

        # Write the DIMACS string to a temporary file and pass it as stdin
        with tempfile.TemporaryFile(mode='w+') as tmp_file:
            tmp_file.write(dimacs_str)
            tmp_file.seek(0)
            result = subprocess.run(call, capture_output=True, text=True, stdin=tmp_file)
    else:

        if var_params is None:
            bin_path = SOLVER_BIN_PATHS["march"]
        else:
            bin_path = SOLVER_BIN_PATHS["march_weighted"]
        call = [f"{bin_path}"]

        tmp_dir = "./data/tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_file_name = os.path.join(tmp_dir, f"tmp_{abs(hash(dimacs_str))}.cnf")
        with open(tmp_file_name, "w") as f:
            f.write(dimacs_str)
        call.append(tmp_file_name)
        result = subprocess.run(call, capture_output=True, text=True, cwd=os.getcwd())
        os.remove(tmp_file_name)

    stats = stdout_to_results_dict(result.stdout)
    return stats
