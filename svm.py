import numpy as np
import cvxpy as cp

from regression import KernelRegression
from kernel import linear_kernel

class SupportVectorMachine(KernelRegression):
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.K is None:
            assert X is not None

            self.K = self.kernel(X, X)
            self.X = X

        self.y = y

        n = len(self.X)
        alpha = cp.Variable(n)

        objective = cp.Minimize(.5 * cp.quad_form(alpha, self.K) - alpha.T @ y)
        alphay = cp.multiply(alpha, y)
        problem = cp.Problem(objective, [
            alphay >= 0,
            alphay <= self.regularization # C
        ])

        problem.solve()
        self.alpha = alpha.value

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from kernel import gaussian_kernel

    n = 10
    X = np.linspace(0, 10, n).reshape([-1, 1])
    y = np.sign(np.sin(X.ravel()) + np.random.randn(n) * .3)
    plt.scatter(X, y)

    x = np.linspace(-1, 11).reshape(-1, 1)

    # Linear kernel
    svm = SupportVectorMachine(gaussian_kernel(.5), regularization=2)
    svm.fit(X, y)
    plt.plot(x, svm.predict(x))

    plt.show()
    