import functools

from numba import njit

from tqdm import tqdm

from kernel.utils import memoize_id


## Substring
@njit()
def substring_kernel_pairwise(s, t, p=2, lam=0.5):
        #see p369 of https://people.eecs.berkeley.edu/~jordan/kernels/0521813972c11_p344-396.pdf
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

        DP = np.zeros((n+1, m+1))
        for l in range(2, p + 1):
            Kern = 0
            for i in range(1, n):
                for j in range(1, m) :
                    DP[i, j] = DPS[i, j] \
                        + lam * (DP[i-1, j] + DP[i, j-1]) \
                        - lam2 * DP[i-1, j-1]
            for i, j in same_letter_indices:
                DPS[i, j] = lam2 * DP[i-1, j-1]
                Kern += DPS[i, j]
            
        return Kern

@functools.lru_cache(None)
def substring_kernel(p=2, lam=0.5):
    @memoize_id
    @njit(parallel=True)
    def substring_kernel_inner(X0: list, X1: list):
        print("hello")
        n0 = len(X0)
        n1 = len(X1)
        K = np.zeros(shape=(n0, n1))

        symmetric = (X0 is X1)
        if symmetric:
            for i in prange(n0):
                for j in prange(i, n1):
                    K[i, j] = substring_kernel_pairwise(X0[i], X1[j], p=p, lam=lam) 
                    K[j, i] = K[i, j]
            return K

        else:
            for i in prange(n0):
                for j in prange(n1):
                    K[i, j] = substring_kernel_pairwise(X0[i], X1[j], p=p, lam=lam)
        
            return K

    return substring_kernel_inner
