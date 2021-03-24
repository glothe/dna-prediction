from datetime import datetime
from load import *
from regression import *
from kernel.spectrum import *
from kernel.mkl import *
from kernel.classic import *
from svm import *

now = str(datetime.now())


def compute_result():
    ## ______ Hyperparamaters ______
    ## MLK
    k_ = 6
    sig2 = 0.0018807171973303132
    alpha = 0.8092075217546304
    C = 8.487642132378188

    k1 = spectrum_kernel(k_)
    k2 = gaussian_kernel(sig2)
    kernel_MLK = weighted_sum_kernel(k1, k2, alpha)

    ## SVM with gaussian_kernel
    # sig2 = np.array([
    #     0.0018549184100468577,
    #     0.003108764735662679,
    #     0.02262574050401125
    # ])

    # C = np.array([
    #     0.6794728013566558,
    #     2.1016123406749423,
    #     1.4808691062181178
    # ])


    ## ______ Load data ______
    Xtr, Xte, Xmattr, Xmatte, ytr = dict(), dict(), dict(), dict(), dict()

    for index in index_list:
        Xtr[index] = load_X(index, mode="train")       # for string kernels
        Xte[index] = load_X(index, mode="test")
        Xmattr[index] = load_Xmat(index, mode="train")    # for numerical kernels
        Xmatte[index] = load_Xmat(index, mode="test")
        ytr[index] = load_y(index)

    ## ______ Predict ______
    yte = dict()

    for index in index_list:
        # svm = SupportVectorMachine(kernel=gaussian_kernel(sig2[index]), regularization=C[index])
        svm = SupportVectorMachine(kernel=kernel_MLK, regularization=C)
        svm.fit((Xtr[index], Xmattr[index]), ytr[index])
        yte[index] = (np.sign(svm.predict((Xte[index], Xmatte[index]))) + 1) / 2
    
    ## ______ Concatenation if needed ______
    if index_list!=[3]:
        yte = np.concatenate(tuple(yte.values()))
    index_column = np.arange(len(yte))
    yte_full = np.concatenate(
        (index_column[:, None], yte[:, None]), 
        axis=1)

    ## ______ Saving result ______
    np.savetxt(f"../data/Yte_{now}.csv", yte_full, fmt="%d", delimiter=',', newline='\n', header='Id,Bound', footer='', comments='', encoding=None)

if __name__ == "__main__":
    index_list = [3]
    # index_list = [0, 1, 2]
    compute_result()
