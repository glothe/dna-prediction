"""
Kernel Ridge/Logistic Regression
"""
import numpy as np
import scipy.spatial.distance

from load import *
from kernel.classic import linear_kernel


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class KernelRegression():
    def __init__(self, kernel=linear_kernel, regularization: float = 0.01):
        self.kernel = kernel
        self.regularization = regularization  # ridge regularization 'lambda'

        self.alpha = None 
        self.K = None
        self.X = None        # X train [samples, features]
        self.y = None        # y train [samples]

    def predict(self, X: np.ndarray = None):
        # X = None -> predict on the training set   
        if X is None:
            K = self.K
        else:
            K = self.kernel(self.X, X)

        return np.dot(self.alpha, K)

    def mse(self, X: np.ndarray = None, y: np.ndarray = None):
        # X = None -> mse on the training set
        y_pred = self.predict(X)

        if y is None:
            y = self.y

        error = y - y_pred
        return np.mean(error * error)

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        return np.mean((y == np.sign(self.predict(X))))
    
    def score(self, X: np.ndarray, y: np.ndarray):
        return self.accuracy(X,y)

class KernelRidgeRegression(KernelRegression):
    ## See slide 94
    def fit(self, X: np.ndarray = None, y: np.ndarray = None, weights: np.ndarray = None):
        if self.K is None:
            assert X is not None

            self.K = self.kernel(X, X)
            self.X = X

        self.y = y

        n = len(y)
        if weights is None:
            A = self.K.copy()
            A.flat[::n+1] += self.regularization * n
            self.alpha = np.linalg.solve(A, y)

        else:
            # See slide 103 for WKRR
            w12 = np.sqrt(weights)
            A = w12[:, None] * self.K * w12[None, :]
            A.flat[::n+1] += self.regularization * n
            self.alpha = w12 * np.linalg.solve(A, w12*y)


class KernelLogisticRegression(KernelRegression):
    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 5):
        if self.K is None:
            assert X is not None

            self.K = self.kernel(X, X)
            self.X = X

        self.y = y

        n = len(y)
        alpha = np.ones(shape=n) / np.sqrt(n)

        for i in range(max_iter):

            ## See slide 114
            m = np.dot(self.K, alpha)
            s = sigmoid(m)
            weights = s * (1 - s)
            z = m + y / sigmoid(y * m)

            w12 = np.sqrt(weights)
            A = w12[:, None] * self.K * w12[None, :]
            A.flat[::n+1] += self.regularization * n
            alpha = w12 * np.linalg.solve(A, w12*z)

        self.alpha = alpha

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from kernel import gaussian_kernel

    n = 10
    X = np.linspace(0, 10, n).reshape([-1, 1])
    y = np.sign(np.sin(X.ravel()) + np.random.randn(n) * .3)
    plt.scatter(X, y)

    x = np.linspace(-1, 11).reshape(-1, 1)

    # Linear kernel
    krr = KernelRidgeRegression(linear_kernel)
    krr.fit(X, y)
    plt.plot(x, krr.predict(x), label="KRR with Linear kernel")

    # Gaussian kernel
    krr = KernelRidgeRegression(gaussian_kernel(.5))
    krr.fit(X, y, weights=X.ravel())
    plt.plot(x, krr.predict(x), label="KRR with Gaussian kernel")

    # Logistic regression
    klr = KernelLogisticRegression(gaussian_kernel(.5))
    klr.fit(X, y)
    plt.plot(x, klr.predict(x), label="KLR with Gaussian kernel")

    plt.legend()
    plt.show()
    