

# Multiple kernel learning
def weighted_sum_kernel(k1, k2, alpha: float = .5):
    def inner(X0: tuple, X1: tuple):
        X0_k1, X0_k2 = X0
        X1_k1, X1_k2 = X1

        return k1(X0_k1, X1_k1) * (1 - alpha) + k2(X0_k2, X1_k2) * alpha

    return inner