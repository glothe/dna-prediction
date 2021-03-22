from datetime import datetime
from load import *
from regression import *
from kernel import *
from svm import *

now = str(datetime.now())

def compute_result():
    ## ______ Hyperparamaters ______
    ## SVM with spectrum_kernel
    # k = np.array([

    # ])

    # C = np.array([

    # ])

    ## SVM with gaussian_kernel
    sig2 = np.array([
        0.0018549184100468577,
        0.003108764735662679,
        0.02262574050401125
    ])

    C = np.array([
        0.6794728013566558,
        2.1016123406749423,
        1.4808691062181178
    ])


    ## ______ Load data ______
    Xtr, Xte, ytr = dict(), dict(), dict()

    for index in [0, 1, 2]:
        # Xtr[index] = load_X(index, mode="train")       # for string kernels
        # Xte[index] = load_X(index, mode="test")
        Xtr[index] = load_Xmat(index, mode="train")    # for numerical kernels
        Xte[index] = load_Xmat(index, mode="test")
        ytr[index] = load_y(index)

    ## ______ Predict ______
    yte = dict()
    for index in [0, 1, 2]:
        # SVM
        # svm = SupportVectorMachine(kernel=spectrum_kernel(k[index]), regularization=C[index])
        svm = SupportVectorMachine(kernel=gaussian_kernel(sig2[index]), regularization=C[index])
        svm.fit(Xtr[index], ytr[index])
        yte[index] = (np.sign(svm.predict(Xte[index])) + 1) / 2
    
    ## ______ Concatenation ______
    # yte = dict()
    # yte[0] = [2,4]
    # yte[1] = [2,4]

    yte = np.concatenate(tuple(yte.values()))
    index_column = np.arange(len(yte))
    yte_full = np.concatenate(
        (index_column[:, None], yte[:, None]), 
        axis=1)

    ## ______ Saving result ______
    np.savetxt(f"../data/Yte_{now}.csv", yte_full, fmt="%d", delimiter=',', newline='\n', header='Id,Bound', footer='', comments='', encoding=None)

if __name__ == "__main__":
    compute_result()
