

from functools import partial

# Debugging
import sklearn.kernel_ridge
import sklearn.linear_model 
import sklearn.svm

# Tuning
import optuna

# Custom
from load import *
from regression import *
from svm import *

from kernel.classic import linear_kernel, gaussian_kernel
from kernel.spectrum import spectrum_kernel
from kernel.mismatch import mismatch_kernel
from kernel.mkl import weighted_sum_kernel

from sklearn.model_selection import KFold

## ______ Parameters ______
index = 1
split_param = .8

Xmat, X, y = load_Xmat(index), load_X(index), load_y(index)

Xmattr, Xmatva = split(Xmat, split_param)
Xtr, Xva = split(X, split_param)
ytr, yva = split(y, split_param)

## ______ All test functions ______
## Sklearn to compare
def sklearn_krr(trial):
    # Best .5525
    sig2 = trial.suggest_float("sig2", 1e-5, 1, log=True)
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)

    krr = sklearn.kernel_ridge.KernelRidge(alpha=.5/C, gamma=1/sig2)
    krr.fit(Xtr, ytr)

    y_pred = krr.predict(Xva)
    return np.mean(yva == np.sign(krr.predict(Xva)))

def sklearn_klr(trial):
    # Best .5525 (idem sklearn_krr ???)
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)

    krr = sklearn.linear_model.LogisticRegression(C=C)
    krr.fit(Xtr, ytr)

    y_pred = krr.predict(Xva)
    return np.mean(yva == y_pred)

def sklearn_svm(trial):
    # Best .5525 (idem sklearn_krr ???)
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)

    svm = sklearn.svm.SVR(kernel="rbf", gamma="auto", C=C)
    svm.fit(Xtr, ytr)

    y_pred = np.sign(svm.predict(Xva))
    return np.mean(yva == y_pred)

## Sanity check
def random():
    n = len(yva)
    y_pred = np.sign(np.random.randn(n))
    err = y_pred - yva
    return np.mean(err * err)

## Linear kernel
def krr_linear(trial):
    lambda_ = trial.suggest_float("lambda_", 1e-3, 1e4, log=True)
    krr = KernelRidgeRegression(kernel=linear_kernel(), regularization=lambda_)
    krr.fit(Xtr, ytr)
    return krr.accuracy(Xva, yva)

def klr_linear(trial):
    lambda_ = trial.suggest_float("lambda_", 1e-3, 1e4, log=True)
    klr = KernelLogisticRegression(kernel=linear_kernel(), regularization=lambda_)
    klr.fit(Xtr, ytr)
    return klr.accuracy(Xva, yva)

def svm_linear(trial):
    C = trial.suggest_float("C", 1e-3, 1e4, log=True)
    svm = SupportVectorMachine(kernel=linear_kernel(), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)

## Gaussian kernel
def krr_gaussian(trial):
    # Best  0.62 (index=0)      {'sig2': 0.00040119005918509015, 'lambda_': 7.671094553679527}/
    #                           {'sig2': 0.0006502378619003336, 'lambda_': 1.8867088604759066}/
    #                           {'sig2': 0.00042733583887551483, 'lambda_': 14.656814090693382}
    #       0.605 (index=1)     {'sig2': 0.0006870026462625262, 'lambda_': 15.512765105839954}
    #                           {'sig2': 0.0006938366890412571, 'lambda_': 16.809662576547957}
    #                           {'sig2': 0.0006869127104215193, 'lambda_': 83.241782506501}
    #                           {'sig2': 0.0006357022892470294, 'lambda_': 99.92776424652297}
    #                           {'sig2': 0.0006637486414259983, 'lambda_': 78.48579339269052}
    #       0.68 (index=2)      {'sig2': 0.0007252171981136182, 'lambda_': 58.49868328302998}
                        
    sig2 = trial.suggest_float("sig2", 1e-4, 1e-3, log=True)
    lambda_ = trial.suggest_float("lambda_", 1e-2, 1e2, log=True)

    krr = KernelRidgeRegression(kernel=gaussian_kernel(sig2), regularization=lambda_)
    krr.fit(Xmattr, ytr)

    return krr.accuracy(Xmatva, yva)

def klr_gaussian(trial):
    # Best  0.62 (index=0)      {'sig2': 0.0006435829108282353, 'lambda_': 14.427768167767454}
    #       0.605 (index=1)     {'sig2': 0.0006942397232263402, 'lambda_': 55.51608382697244}
    #       0.68 (index=2)      {'sig2': 0.000726386515015593, 'lambda_': 11.30392930219456}

    sig2 = trial.suggest_float("sig2", 1e-4, 10, log=True)
    lambda_ = trial.suggest_float("lambda_", 1, 1e2, log=True)

    klr = KernelLogisticRegression(kernel=gaussian_kernel(sig2), regularization=lambda_)
    klr.fit(Xmattr, ytr)

    return klr.accuracy(Xmatva, yva)

def svm_gaussian(trial):
    # Best  0.62 (index=0)     {'sig2': 0.0004475252677121083, 'C': 3.7371130579973366} #same   with {'sig2': 0.0005929692706695875, 'C': 2.2862830866907577}
    #       0.625 (index=1)    {'sig2': 0.026348581508309914, 'C': 3.521226987220084}   #6225   with {'sig2': 0.01877452515739, 'C': 1.7176296305911665}
    #       0.735 (index=2)    {'sig2': 0.007605991116858567, 'C': 0.8547649197628695}  #0.7275 with {'sig2': 0.006672040107687773, 'C': 1.0330032957410673}

    # TODO Check convergence
    sig2 = trial.suggest_float("sig2", 1e-5, 1e-1, log=True)
    C = trial.suggest_float("C", 1e-2, 10, log=True)

    svm = SupportVectorMachine(kernel=gaussian_kernel(sig2), regularization=C)
    svm.fit(Xmattr, ytr)
    return svm.accuracy(Xmatva, yva)

def svm_gaussian_cv(trial):
    # Best  0.6255 (index=0)   {'sig2': 0.0018549184100468577, 'C': 0.6794728013566558}
    #       0.603 (index=1)    {'sig2': 0.003108764735662679, 'C': 2.1016123406749423}
    #       0.711 (index=2)    {'sig2': 0.02262574050401125, 'C': 1.4808691062181178}

    sig2 = trial.suggest_float("sig2", 1e-5, 1e-1, log=True)
    C = trial.suggest_float("C", 1e-2, 10, log=True)

    total_score = 0.
    kf5 = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf5.split(range(len(X))):
        svm = SupportVectorMachine(kernel=gaussian_kernel(sig2), regularization=C)
        svm.fit(Xmat[train_index], y[train_index])
        score = svm.accuracy(Xmat[test_index], y[test_index])
        print(score)
        total_score += score
    
    return total_score / 5

# Spectrum
def svm_spectrum(trial):
    # Best  0.6300 (index=0)    
    #       0.6175 (index=1) {'C': 7.77391663055827, 'k': 4}
    #       0.6975 (index=2)   
    C = trial.suggest_float("C", 1e-1, 1e2, log=True)
    k = trial.suggest_int("k", 3, 5)

    svm = SupportVectorMachine(kernel=spectrum_kernel(k), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)

# Mismatch - not pd yet
def klr_mismatch(trial): #Does not work
    C = trial.suggest_float("C", 1, 1e3, log=True)
    k = trial.suggest_int("k", 3, 5)
    m = trial.suggest_int("m", 1, 2)

    klr = KernelLogisticRegression(kernel=mismatch_kernel(k, m), regularization=C)
    klr.fit(Xtr, ytr)
    return klr.accuracy(Xva, yva)

def svm_mismatch(trial):
    C = trial.suggest_float("C", 1, 1e3, log=True)
    k = trial.suggest_int("k", 3, 5)
    m = trial.suggest_int("m", 1, 2)

    svm = SupportVectorMachine(kernel=mismatch_kernel(k, m), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)

# MKL
def svm_spectrum_gaussian(trial):
    # Best  0
    #       0.6325 (index=1)    {'sig2': 0.02071621910845629, 'alpha': 0.8469436560455597, 'C': 2.143111748529303}
    #       0
    sig2 = trial.suggest_float("sig2", 1e-5, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 0, 1)
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)

    k1 = spectrum_kernel(4)
    k2 = gaussian_kernel(sig2)
    kernel = weighted_sum_kernel(k1, k2, alpha)

    # svm = SupportVectorMachine(kernel=kernel, regularization=C)
    svm = KernelLogisticRegression(kernel=kernel, regularization=C)
    svm.fit((Xtr, Xmattr), ytr)
    return svm.accuracy((Xva, Xmatva), yva)


# Substring - too slow
def svm_substring(trial):
    C = trial.suggest_float("C", 1e-1, 1e2, log=True)
    p = 5 #trial.suggest_int("p", 2, 2)
    lam = 0.9 # trial.suggest_float("lam", 0, 1)

    svm = SupportVectorMachine(kernel=substring_kernel(p, lam), regularization=C)
    svm.fit(Xtr[:200], ytr[:200])
    return svm.accuracy(Xva[:50], yva[:50])



if __name__ == "__main__":
    print("_______ Dataset index=", index)
    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(svm_spectrum_gaussian, n_trials=100, n_jobs=-1)

    except KeyboardInterrupt:
        pass

    # from optuna.visualization import plot_contour
    # fig = plot_contour(study, params=["C", "sig2"])
    # fig.show()