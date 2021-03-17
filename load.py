import os.path
import csv

import numpy as np

MODE_IDENTIFIER = {
    "train": "tr",
    "test": "te"
}

def load_X(index=0, mode="train", folder="../data"):
    """Loads the sequence files"""
    mode_identifier = MODE_IDENTIFIER[mode]
    fn = os.path.join(folder, f"X{mode_identifier}{index}.csv")
    print(f"Loading {fn}.")

    sequences = []
    with open(fn, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["seq"]
            sequences.append(seq)

    return tuple(sequences)

def load_Xmat(index=0, mode="train", folder="../data"):
    """Loads the simplified sequence files"""
    mode_identifier = MODE_IDENTIFIER[mode]
    fn = os.path.join(folder, f"X{mode_identifier}{index}_mat100.csv")
    print(f"Loading {fn}.")
    return np.genfromtxt(fn, dtype=float)

def load_y(index=0, folder="../data"):
    """Loads the label files"""
    fn = os.path.join(folder, f"Ytr{index}.csv")
    print(f"Loading {fn}.")
    return np.genfromtxt(fn, skip_header=1, usecols=[1], delimiter=",", dtype=float) * 2 - 1

def split(data, frac=.8):
    n = int(len(data) * frac)
    return data[:n], data[n:]


if __name__ == "__main__":
    load_X()