from pysat.formula import CNF


def load_dimacs_cnf(path: str) -> list[list[int]] | None:
    """
    Loads a CNF formula from a file in DIMACS CNF format.
    Supports both uncompressed (.cnf) and compressed (.cnf.xz) files.

    :param path: The path to a .cnf or .cnf.xz file in DIMACS format.
    :return: A list of clauses, each clause is a list of integers.
    """
    return CNF(from_file=path).clauses


def write_dimacs_cnf(f: list[list[int]], path: str) -> None:
    """
    Stores a cnf formula in the dimacs cnf format
    :param f: The formula as a list of lists of signed integers.
    :param path: The path to a file in which f is will be stored
    """
    CNF(from_clauses=f).to_file(path)
