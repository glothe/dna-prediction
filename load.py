"""
Load the data files for training and testing and option to split them
"""

import os.path
import csv
from itertools import product

import numpy as np

MODE_IDENTIFIER = {
    "train": "tr",
    "test": "te"
}

DINUCLEOTIDE_NAMES = "BCDEFHIJKLMNOPQR"
DINUCLEOTIDES = {"".join(di):DINUCLEOTIDE_NAMES[i] for i, di in enumerate(product("ATCG", repeat=2))}

def load_X(index=0, mode="train", folder="../data"):
    """
    Loads the raw sequence files
    """
    mode_identifier = MODE_IDENTIFIER[mode]
    fn = os.path.join(folder, f"X{mode_identifier}{index}.csv")
    print(f"Loading {fn}.")

    sequences = []
    with open(fn, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["seq"]
            sequences.append(seq)

    return np.array(sequences, dtype="str")

def load_Xdi(index=0, mode="train", folder="../data"):
    """
    Loads the sequence files assembled into di-nucleotides
    """
    X = load_X(index, mode, folder)
    Xdi = np.zeros_like(X)

    for i, x in enumerate(X):
        chain = []
        for j in range(len(x) - 1):
            chain.append(DINUCLEOTIDES[x[j:j + 2]])
        Xdi[i] = "".join(chain)
        
    return Xdi

def load_Xmat(index=0, mode="train", folder="../data"):
    """
    Loads the simplified sequence files
    """
    mode_identifier = MODE_IDENTIFIER[mode]
    fn = os.path.join(folder, f"X{mode_identifier}{index}_mat100.csv")
    print(f"Loading {fn}.")
    return np.genfromtxt(fn, dtype=float)

def load_y(index=0, folder="../data"):
    """
    Loads the label files
    """
    fn = os.path.join(folder, f"Ytr{index}.csv")
    print(f"Loading {fn}.")
    return np.genfromtxt(fn, skip_header=1, usecols=[1], delimiter=",", dtype=float) * 2 - 1

def split(data, frac=.8):
    """
    Deterministic split
    """
    n = int(len(data) * frac)
    return data[:n], data[n:]


if __name__ == "__main__":
    load_X()
    print(load_Xdi())