import functools
from collections import defaultdict, Counter
from itertools import combinations, product

import numpy as np

from tqdm import tqdm

from kernel.utils import memoize_id


## Mismatch kernel
def get_mismatch(substr: str, list_index: list, alphabet: str ="ATCG"):
    """ Get all words similar to 'substr' with mismatch characters taken from 'alphabet'
    in the locations defined by 'list_index'
    """
    replacement_letters_list = list(product(alphabet, repeat=len(list_index)))
    l = []
    for i, letters in enumerate(replacement_letters_list):
        s = list(substr)
        for j, index in enumerate(list_index):
            s[index] = letters[j]
        s = ''.join(s)
        l.append(s)
    return l

@functools.lru_cache(None)
#@functools.lru_cache(2**20)
def get_mismatch_list(substr: str, m: int) -> set:
    """ Get all mismatch words of 'substr' up to 'm' mismatchs
    """
    s = set()
    indices = combinations(range(len(substr)), m)
    for index_i in indices:
        mismatch_i = get_mismatch(substr, list(index_i))
        s.update(mismatch_i)
    return s

    d_numba = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64,
    )
    for key, value in d.items():
        d_numba[key] = value

    return d_numba

@functools.lru_cache(None)
def mismatch_kernel(m: int, k: int):
    """Substrings of len k with at most m mismatches"""
    @memoize_id
    def mismatch_kernel_inner(X0: list, X1: list):
        n0 = len(X0)
        n1 = len(X1)
        K = np.zeros(shape=(n0, n1))

        symmetric = X0 is X1

        # Build X0_dict
        X0_dict = [None] * n0 
        for i, x0 in tqdm(enumerate(X0), total=n0, desc=f"Mismatch kernel init (k={k}, m={m})"):
            X0_dict[i] = defaultdict(int)
            substr_list = Counter(x0[c:c+k] for c in range(len(x0)-k+1)).keys()
            for c in range(len(x0)-k+1):
                for mismatch_word in get_mismatch_list(x0[c:c+k], m):
                    if mismatch_word in substr_list:
                        X0_dict[i][mismatch_word] += 1

        # Build X1_dict
        if symmetric:
            X1_dict = X0_dict   
        else:
            X1_dict = [None] * n1
            for i, x1 in enumerate(X1):
                substr_list = Counter(x1[c:c+k] for c in range(len(x1)-k+1)).keys()
                X1_dict[i] = defaultdict(int)
                for c in range(len(x1)-k+1):
                    for mismatch_word in get_mismatch_list(x1[c:c+k], m):
                        if mismatch_word in substr_list:
                            X1_dict[i][mismatch_word] += 1
        
        # Compute K
        if symmetric:
            for i in tqdm(range(n0), desc=f"Mismatch kernel (k={k}, m={m})"):
                X0i = X0_dict[i]
                for j in range(i, n1):
                    X1j = X1_dict[j]
                    K[i, j] = sum(count * X1j[substr] for substr, count in X0i.items())
                    K[j, i] = K[i, j]
        else:
            for i in tqdm(range(n0), desc=f"Mismatch kernel (k={k}, m={m})"):
                X0i = X0_dict[i]
                for j in range(n1):
                    X1j = X1_dict[j]
                    K[i, j] = sum(count * X1j[substr] for substr, count in X0i.items())

        return K

    return mismatch_kernel_inner

