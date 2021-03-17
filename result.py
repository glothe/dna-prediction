from datetime import datetime
from load import *
from regression import *
from kernel import *
from svm import *

now = str(datetime.now())

def compute_result():
    ## Hyperparamaters for line 27 #SupportVectorMachine(kernel=spectrum_kernel(m), regularization=C)
    m = 4
    C = 12

    ## Load data
    Xtr, Xte, ytr = dict(), dict(), dict()

    for index in [0, 1, 2]:
        Xtr[index] = load_X(index, mode="train")       # for string kernels
        Xte[index] = load_X(index, mode="test")
        # Xtr = load_Xmat(index, mode="train")    # for numerical kernels
        # Xte = load_Xmat(index, mode="test")
        ytr[index] = load_y(index)

    ## Predict
    yte = dict()
    for index in [0, 1, 2]:
        # SVM
        svm = SupportVectorMachine(kernel=spectrum_kernel(m), regularization=C)
        svm.fit(Xtr[index], ytr[index])
        yte[index] = (np.sign(svm.predict(Xte[index])) + 1) / 2
    
    ## Concatenation
    yte = np.concatenate(tuple(yte.values()))
    index_column = np.arange(len(yte))
    yte_full = np.concatenate(
        (index_column[:, None], yte[:, None]), 
        axis=1)

    ## Saving result
    np.savetxt(f"../data/Yte_{now}.csv", yte_full, fmt="%d", delimiter=',', newline='\n', header='Id,Bound', footer='', comments='', encoding=None)

if __name__ == "__main__":
    compute_result()
