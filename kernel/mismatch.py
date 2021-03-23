import functools
from collections import defaultdict
from itertools import combinations, product

import numpy as np

from tqdm import tqdm

from kernel.utils import memoize_id


ALPHABET = "ATCG"


# Mismatch String Kernels for SVM Protein Classification
# https://noble.gs.washington.edu/papers/leslie_mismatch.pdf

def get_mismatch(substr: str, list_index: list):
    """ 
    Get all words similar to 'substr' with mismatch in the locations defined by 'list_index'
    """
    l = []
    for i, letters in enumerate(product(ALPHABET, repeat=len(list_index))):
        s = list(substr)
        for j, index in enumerate(list_index):
            s[index] = letters[j]
        s = ''.join(s)
        l.append(s)
    return l

@functools.lru_cache(None)
def get_mismatch_list(substr: str, m: int) -> set:
    """ 
    Get all mismatch words of 'substr' up to 'm' mismatchs
    """
    s = set()
    indices = combinations(range(len(substr)), m)
    for index_i in indices:
        mismatch_i = get_mismatch(substr, list(index_i))
        s.update(mismatch_i)
    return s

@memoize_id
def feature_vector(X: np.ndarray, k: int = 4, m: int = 2):
    n = len(X)
    X_dict = [None] * n
    for i, x in enumerate(tqdm(X, desc=f"Mismatch kernel feature vector (k={k}, m={m})")):
        X_dict[i] = defaultdict(int)

        for c in range(len(x) - k + 1):
            for mismatch in get_mismatch_list(x[c:c+k], m):
                X_dict[i][mismatch] += 1
        
    return X_dict


@functools.lru_cache(None)
def mismatch_kernel(k: int = 4, m: int = 2):
    """
    Substrings of len k with at most m mismatches
    """

    @memoize_id
    def mismatch_kernel_inner(X0: list, X1: list):
        # Same as spectrum kernel basically with a different feature vector
        symmetric = X0 is X1

        n0 = len(X0)
        X0_dict = feature_vector(X0, k, m)

        if symmetric:
            K = np.zeros(shape=(n0, n0))
            for i in tqdm(range(n0), desc=f"Mismatch kernel (k={k}, m={m})"):
                X0i = X0_dict[i]
                for j in range(i, n0):
                    X0j = X0_dict[j]
                    K[i, j] = sum(count * X0j[substr] for substr, count in X0i.items())
                    K[j, i] = K[i, j]

        else:
            n1 = len(X1)
            X1_dict = feature_vector(X1, k, m)

            K = np.zeros(shape=(n0, n1))
            for i in tqdm(range(n0), desc=f"Mismatch kernel (k={k}, m={m})"):
                X0i = X0_dict[i]
                for j in range(n1):
                    X1j = X1_dict[j]
                    K[i, j] = sum(count * X1j[substr] for substr, count in X0i.items())

        return K

    return mismatch_kernel_inner


if __name__ == "__main__":
    from load import *

    X = load_X()

    print(mismatch_kernel()(X, X))