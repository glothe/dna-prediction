import numpy as np
import scipy.spatial.distance

from collections import Counter, defaultdict
from itertools import combinations, product
import functools

from tqdm import tqdm

from numba import njit, prange


## Utils for Spectrum and Substring kernel
def memoize_id(func):
    """If already computed, does not recompute the kernel result"""
    # memoize by object id
    func.cache = {}

    def wrapper(*args):
        id_ = tuple([id(arg) for arg in args])
        if id_ not in func.cache:
            func.cache[id_] = func(*args)
        return func.cache[id_]

    return wrapper

## Spectrum kernel
@functools.lru_cache(None)
def spectrum_kernel(k=4):
    @memoize_id
    def spectrum_kernel_inner(X0: list, X1: list):
        n0 = len(X0)
        n1 = len(X1)
        K = np.zeros(shape=(n0, n1))

        symmetric = X0 is X1

        # Build X0_dict
        X0_dict = [None] * n0 
        for i, x0 in enumerate(X0):
            X0_dict[i] = Counter(x0[c:c+k] for c in range(len(x0)-k+1))

        # Build X1_dict
        if symmetric:
            X1_dict = X0_dict   
        else:
            X1_dict = [None] * n1
            for i, x1 in enumerate(X1):
                X1_dict[i] = Counter(x1[c:c+k] for c in range(len(x1)-k+1))
        # Compute K
        if symmetric:
            for i in tqdm(range(n0), desc=f"Spectrum kernel (k={k})"):
                X0i = X0_dict[i]
                for j in range(i, n1):
                    X1j = X1_dict[j]
                    K[i, j] = sum(count * X1j[substr] for substr, count in X0i.items())
                    K[j, i] = K[i, j]
        else:
            for i in tqdm(range(n0), desc=f"Spectrum kernel (k={k})"):
                X0i = X0_dict[i]
                for j in range(n1):
                    X1j = X1_dict[j]
                    K[i, j] = sum(count * X1j[substr] for substr, count in X0i.items())

        return K

    return spectrum_kernel_inner

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

        # print(X0_dict[0])

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

        # print(X1_dict[0])
        
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


## Substring
@njit()
def substring_kernel_pairwise(s, t, p=2, lam=0.5):
        #see p369 of https://people.eecs.berkeley.edu/~jordan/kernels/0521813972c11_p344-396.pdf
        # assert 0 <= lam and lam <= 1
        # assert p>1
        n = len(s)
        m = len(t)
        DPS = np.zeros((n+1, m+1)) #shifted to have a column and line of zeros
        lam2 = lam*lam

        same_letter_indices = []
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s[i-1] == t[j-1]:
                    same_letter_indices.append((i, j))
                    DPS[i, j] = lam2
        # print("DPS\n", DPS[1:, 1:])

        DP = np.zeros((n+1, m+1))
        for l in range(2, p + 1):
            # print(l)
            Kern = 0
            for i in range(1, n):
                for j in range(1, m) :
                    DP[i, j] = DPS[i, j] \
                        + lam * (DP[i-1, j] + DP[i, j-1]) \
                        - lam2 * DP[i-1, j-1]
                    # if s[i - 1] == t[j - 1]:
                    #     DPS[i, j] = lam2 * DP[i-1, j-1]
                    #     Kern += DPS[i, j]
            for i, j in same_letter_indices:
                DPS[i, j] = lam2 * DP[i-1, j-1]
                Kern += DPS[i, j]

            # print("DP\n", DP[1:, 1:])
            # print()
            # print("DPS\n", DPS[1:, 1:])
            # print()
            
        return Kern

@functools.lru_cache(None)
def substring_kernel(p=2, lam=0.5):
    @memoize_id
    @njit(parallel=True)
    # @njit()
    def substring_kernel_inner(X0: list, X1: list):
        print("hello")
        n0 = len(X0)
        n1 = len(X1)
        K = np.zeros(shape=(n0, n1))

        symmetric = (X0 is X1)
        if symmetric:
            for i in prange(n0):
                # print("sym", i)
            # for i in tqdm(range(n0), desc=f"Substring kernel (p={p}, lam={lam})"):
                for j in prange(i, n1):
                    K[i, j] = substring_kernel_pairwise(X0[i], X1[j], p=p, lam=lam) 
                    K[j, i] = K[i, j]
            return K

        else:
            for i in prange(n0):
                # print("not sym", i)
            # for i in tqdm(range(n0), desc=f"Substring kernel (p={p}, lam={lam})"):
                for j in prange(n1):
                    K[i, j] = substring_kernel_pairwise(X0[i], X1[j], p=p, lam=lam)
        
            return K

    return substring_kernel_inner

## Linear
def linear_kernel():
    def linear_kernel_inner(X0: np.ndarray, X1: np.ndarray):
        return np.dot(X0, X1.T) 
    return linear_kernel_inner

## Gaussian
def gaussian_kernel(sig2: float = 1):
    def gaussian_kernel_inner(X0: np.ndarray, X1: np.ndarray):
        dist = scipy.spatial.distance.cdist(X0, X1, metric="sqeuclidean")
        return np.exp(- 0.5 / sig2 * dist)    
    return gaussian_kernel_inner


if __name__ == "__main__":
    from load import *
    X = load_X()[:1000]
    
    # print(substring_kernel_pairwise("cata", "gatta", p=3, lam=0.1))
    # print(substring_kernel(p=5, lam=0.5)(X[:200], X[:200]))
    #print(substring_kernel_pairwise("cata", "gatta", p=1, lam=0.1))

    ## ________ Positive definite test ________
    # Ks = spectrum_kernel(3)(X, X)
    # print(Ks)
    # Km = mismatch_kernel(2, 5)(X, X)      #1:07
    Km = mismatch_kernel(2, 5)(X, X)        #0:51 with pypy env
    print(Km)
    # print(np.all(np.linalg.eigvals(Ks) >= 0))
    print(np.all(np.linalg.eigvals(Km) >= 0))

    ## ________ gatta cata test ________
    print(spectrum_kernel(2)(["gatta"], ["cata"])) #Returns 2
    print(mismatch_kernel(1, 2)(["GATTA"], ["CATA"])) #Returns 8


