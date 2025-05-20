import os
import signal
import subprocess
import numpy as np


CADICAL_BIN_PATH = "../solvers/cadical/cadical"
DRATTRIM_BIN_PATH = "../solvers/cadical/drat-trim"


def extract_num_variables(cnf_file_path):
    with open(cnf_file_path, 'r') as file:
        for line in file:
            if line.startswith('p cnf'):
                parts = line.strip().split()
                return int(parts[2])
        raise ValueError("No 'p cnf' line found in the file.")


def get_core_vars(cnf_filepath: str, cache: bool = False, keep_aux_files: bool = False):
    if cache:
        cache_file = cnf_filepath + '.core_vars.npy'
        if os.path.exists(cache_file):
            return np.load(cache_file)
    else:
        cache_file = None

    n_vars = extract_num_variables(cnf_filepath)
    core_vars = np.zeros(n_vars)

    proof_filepath = cnf_filepath + ".proof"
    core_filepath = cnf_filepath + ".core"

    if not os.path.exists(proof_filepath):
        prover_cmd_line = [CADICAL_BIN_PATH, "--unsat", cnf_filepath, proof_filepath]
        process = subprocess.Popen(prover_cmd_line, start_new_session=True, stdout=subprocess.DEVNULL)
        process.communicate()

    if not os.path.exists(proof_filepath):
        return core_vars

    if not os.path.exists(core_filepath):
        checker_cmd_line = [DRATTRIM_BIN_PATH, cnf_filepath, proof_filepath, '-c', core_filepath]

        try:
            process = subprocess.Popen(checker_cmd_line, start_new_session=True, stdout=subprocess.DEVNULL)
            process.communicate()
        except:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    if not os.path.exists(core_filepath):
        return core_vars

    with open(core_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            tokens = line.strip().split()
            core_vars[[abs(int(t)) - 1 for t in tokens[:-1]]] = 1

    if cache:
        np.save(cache_file, core_vars)

    if not keep_aux_files:
        os.remove(proof_filepath)
        os.remove(core_filepath)

    return core_vars


if __name__ == '__main__':
    cnf_path = "../../g4satbench/easy/k-clique/test/sat/00003.cnf"
    core_vars = get_core_vars(cnf_path)
    print(core_vars)
