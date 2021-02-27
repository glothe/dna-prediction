import numpy as np
import scipy.spatial.distance

from collections import Counter
import functools

from tqdm import tqdm


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

@functools.lru_cache(None)
def spectrum_kernel(m=4):
    # m=3: 25s
    # m=4: 52s

    @memoize_id
    def spectrum_kernel_inner(X0: list, X1: list):
        n0 = len(X0)
        n1 = len(X1)
        K = np.zeros(shape=(n0, n1))

        symmetric = X0 is X1

        # Build X0_dict
        X0_dict = [None] * n0 
        for i, x0 in enumerate(X0):
            X0_dict[i] = Counter(x0[c:c+m] for c in range(len(x0)-m+1))

        # Build X1_dict
        if symmetric:
            X1_dict = X0_dict   
        else:
            X1_dict = [None] * n1
            for i, x1 in enumerate(X1):
                X1_dict[i] = Counter(x1[c:c+m] for c in range(len(x1)-m+1))
        # Compute K
        if symmetric:
            for i in tqdm(range(n0), desc=f"Spectrum kernel (m={m})"):
                X0i = X0_dict[i]
                for j in range(i, n1):
                    X1j = X1_dict[j]
                    K[i, j] = sum(count * X1j[substr] for substr, count in X0i.items())
                    K[j, i] = K[i, j]
        else:
            for i in tqdm(range(n0), desc=f"Spectrum kernel (m={m})"):
                X0i = X0_dict[i]
                for j in range(n1):
                    X1j = X1_dict[j]
                    K[i, j] = sum(count * X1j[substr] for substr, count in X0i.items())

        return K

    return spectrum_kernel_inner


def linear_kernel(X0: np.ndarray, X1: np.ndarray):
    return np.dot(X0, X1.T) 

def gaussian_kernel(sig2: float = 1):
    def gaussian_kernel_inner(X0: np.ndarray, X1: np.ndarray):
        dist = scipy.spatial.distance.cdist(X0, X1, metric="sqeuclidean")
        return np.exp(- 0.5 / sig2 * dist)    
    return gaussian_kernel_inner


if __name__ == "__main__":
    from load import *

    X = load_X()
    
    spectrum_kernel(3)(X, X)