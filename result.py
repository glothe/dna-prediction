"""
Run this file to reproduce our final Kaggle submission
"""

from datetime import datetime
from load import *
from regression import *
from kernel.spectrum import *
from kernel.mkl import *
from kernel.classic import *
from svm import *

now = str(datetime.now())


def compute_result(index_list):
    """ Compute Yte.csv to reproduce our bert submission
    index_list = [3] to use all data together for prediction
    """
    ## ______ Hyperparamaters ______
    ## Final results, using spectrum_kernel and gaussian_kernel
    ## Parameters were obtained using tune.py
    k_ = 6
    sig2 = 0.0018807171973303132
    alpha = 0.8092075217546304
    C = 8.487642132378188

    k1 = spectrum_kernel(k_)
    k2 = gaussian_kernel(sig2)
    kernel = weighted_sum_kernel(k1, k2, alpha)


    ## ______ Load data ______
    Xtr, Xte, Xmattr, Xmatte, ytr = dict(), dict(), dict(), dict(), dict()

    for index in index_list:
        Xtr[index] = load_X(index, mode="train")            # for string kernels
        Xte[index] = load_X(index, mode="test")
        Xmattr[index] = load_Xmat(index, mode="train")      # for numerical kernels
        Xmatte[index] = load_Xmat(index, mode="test")
        ytr[index] = load_y(index)

    ## ______ Predict ______
    yte = dict()

    for index in index_list:
        svm = SupportVectorMachine(kernel=kernel, regularization=C)
        svm.fit((Xtr[index], Xmattr[index]), ytr[index])
        yte[index] = (np.sign(svm.predict((Xte[index], Xmatte[index]))) + 1) / 2
    
    ## ______ Concatenation if needed ______
    yte = np.concatenate(tuple(yte.values()))
    index_column = np.arange(len(yte))
    yte_full = np.concatenate(
        (index_column[:, None], yte[:, None]), 
        axis=1)

    ## ______ Saving result ______
    np.savetxt(f"../data/Yte.csv", yte_full, fmt="%d", delimiter=',', newline='\n', header='Id,Bound', footer='', comments='', encoding=None)

if __name__ == "__main__":
    index_list = [3] #Only using the concatenated file with the three datasets combined
    compute_result(index_list)
