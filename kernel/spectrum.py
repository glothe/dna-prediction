import functools
from collections import Counter

import numpy as np
from numba import njit
from numba.typed import Dict

from tqdm import tqdm

from kernel.utils import memoize_id


TRANSLATION = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C"
}


@functools.lru_cache(None)
def complement(x: str):
    if x[0] in "AC":
        return x
    return x.translate(TRANSLATION)

@memoize_id
def feature_vector(X: np.ndarray, k: int):
    n = len(X)
    X_dict = [None] * n

    for i, x in enumerate(X):
        X_dict[i] = Counter(complement(x[c:c + k]) for c in range(len(x) - k + 1))
        
    return X_dict


@functools.lru_cache(None)
def spectrum_kernel(k: int = 4):
    @memoize_id
    def spectrum_kernel_inner(X0: np.ndarray, X1: np.ndarray):
        symmetric = X0 is X1

        n0 = len(X0)
        X0_dict = feature_vector(X0, k)

        if symmetric:
            K = np.zeros(shape=(n0, n0))

            # Compute sparse dot product
            for i in tqdm(range(n0), desc=f"Spectrum kernel (k={k})"):
                X0i = X0_dict[i]
                for j in range(i, n0):
                    X0j = X0_dict[j]
                    K[i, j] = sum(count * X0j[substr] for substr, count in X0i.items())
                    K[j, i] = K[i, j]

        else:
            n1 = len(X1)
            X1_dict = feature_vector(X1, k)

            K = np.zeros(shape=(n0, n1))

            # Compute sparse dot product
            for i in tqdm(range(n0), desc=f"Spectrum kernel (k={k})"):
                X0i = X0_dict[i]
                for j in range(n1):
                    X1j = X1_dict[j]
                    K[i, j] = sum(count * X1j[substr] for substr, count in X0i.items())

        return K

    return spectrum_kernel_inner