import numpy as np
import scipy.spatial.distance


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
