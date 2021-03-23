import functools

from numba import njit

from tqdm import tqdm

from utils import memoize_id


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


if __name__ == "__main__":
    from load import *
    X = load_X()[:10]
    
    # print(substring_kernel_pairwise("cata", "gatta", p=3, lam=0.1))
    # print(substring_kernel(p=5, lam=0.5)(X[:200], X[:200]))
    #print(substring_kernel_pairwise("cata", "gatta", p=1, lam=0.1))

    ## ________ Positive definite test ________
    # Ks = spectrum_kernel(3)(X, X)
    # print(Ks)
    # Km = mismatch_kernel(2, 5)(X, X)      #1:07
    # Km = spectrum_kernel(1)(X, X)        #0:51 with pypy env

    # ## ________ gatta cata test ________
    # print(spectrum_kernel(2)(["gatta"], ["cata"])) #Returns 2
    # print(mismatch_kernel(1, 2)(["GATTA"], ["CATA"])) #Returns 8


