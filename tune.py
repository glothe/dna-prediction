"""
Tuning of hyperparameters for different kernels with optuna
"""

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
index = 0
split_param = .8

Xmat = load_Xmat(index)
X = load_X(index)
Xdi = load_Xdi(index)
y =load_y(index)

Xmattr, Xmatva = split(Xmat, split_param)
Xtr, Xva = split(X, split_param)
Xditr, Xdiva = split(Xdi, split_param)
ytr, yva = split(y, split_param)

## Cross validation
Xmat_tr_cv, X_tr_cv, y_tr_cv = [],[],[]
Xmat_te_cv, X_te_cv, y_te_cv = [],[],[]

kf5 = KFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in kf5.split(range(len(X))):
    Xmat_tr_cv.append(Xmat[train_index])
    X_tr_cv.append(X[train_index])
    y_tr_cv.append(y[train_index])

    Xmat_te_cv.append(Xmat[test_index])
    X_te_cv.append(X[test_index])
    y_te_cv.append(y[test_index])


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
    # Best .5525
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)

    krr = sklearn.linear_model.LogisticRegression(C=C)
    krr.fit(Xtr, ytr)

    y_pred = krr.predict(Xva)
    return np.mean(yva == y_pred)

def sklearn_svm(trial):
    # Best .5525
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
    krr.fit(Xmattr, ytr)
    return krr.accuracy(Xmatva, yva)

def klr_linear(trial):
    lambda_ = trial.suggest_float("lambda_", 1e-3, 1e4, log=True)
    klr = KernelLogisticRegression(kernel=linear_kernel(), regularization=lambda_)
    klr.fit(Xmattr, ytr)
    return klr.accuracy(Xmatva, yva)

def svm_linear(trial):
    #0.5725
    C = trial.suggest_float("C", 1e-3, 1e4, log=True)
    svm = SupportVectorMachine(kernel=linear_kernel(), regularization=C)
    svm.fit(Xmattr, ytr)
    return svm.accuracy(Xmatva, yva)

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

## Spectrum
def svm_spectrum(trial):
    # Best  0.6300 (index=0)    
    #       0.6175 (index=1) {'C': 7.77391663055827, 'k': 4}
    #       0.6975 (index=2)   
    C = trial.suggest_float("C", 1e-1, 1e2, log=True)
    k = trial.suggest_int("k", 3, 5)

    svm = SupportVectorMachine(kernel=spectrum_kernel(k), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)

## Mismatch
def klr_mismatch(trial):
    C = trial.suggest_float("C", 1, 1e2, log=True)
    k = trial.suggest_int("k", 3, 5)
    m = trial.suggest_int("m", 1, 2)

    klr = KernelLogisticRegression(kernel=mismatch_kernel(k, m), regularization=C)
    klr.fit(Xtr, ytr)
    return klr.accuracy(Xva, yva)

def svm_mismatch(trial):
    C = trial.suggest_float("C", 1, 1e2, log=True)
    k = trial.suggest_int("k", 3, 5)
    m = trial.suggest_int("m", 1, 1)

    svm = SupportVectorMachine(kernel=mismatch_kernel(k, m), regularization=C)
    svm.fit(Xditr, ytr)
    return svm.accuracy(Xdiva, yva)

## Weighted sum of kernels
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

    svm = SupportVectorMachine(kernel=kernel, regularization=C)
    svm.fit((Xtr, Xmattr), ytr)
    return svm.accuracy((Xva, Xmatva), yva)

def svm_spectrum_gaussian_cv(trial):
    # Best  0.66867 (index=3) k=6 {'sig2': 0.0018807171973303132, 'alpha': 0.8092075217546304, 'C': 8.487642132378188} # {'sig2': 0.0070692362862250484, 'alpha': 0.16716878829043003, 'C': 0.9562507635122345}
    #       0.6606 (index=3) k=7  {'sig2': 0.0023952930891851646, 'alpha': 0.5924630838851894, 'C': 0.9496619536392173} #{'sig2': 0.0038072413901906587, 'alpha': 0.5465019523098097, 'C': 0.7315160001875735} #{'sig2': 0.004411729468212926, 'alpha': 0.515781529441073, 'C': 0.586026133218107} # {'sig2': 0.0008152106908624851, 'alpha': 0.7629900839727197, 'C': 2.426485272234043}

    sig2 = trial.suggest_float("sig2", 1e-5, 1e-1, log=True)
    # k_ = trial.suggest_int("k", 3, 5)
    k_ = 7
    alpha = trial.suggest_float("alpha", 0, 1)
    C = trial.suggest_float("C", 1e-2, 10, log=True)
    
    k1 = spectrum_kernel(k_)
    k2 = gaussian_kernel(sig2)
    kernel = weighted_sum_kernel(k1, k2, alpha)

    total_score = 0.

    for i in range(5):
        svm = SupportVectorMachine(kernel=kernel, regularization=C)
        svm.fit((X_tr_cv[i], Xmat_tr_cv[i]), y_tr_cv[i])
        score = svm.accuracy((X_te_cv[i], Xmat_te_cv[i]), y_te_cv[i])
        print(score)
        total_score += score

    return total_score / 5

## Substring - too slow !
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
        study.optimize(svm_spectrum_gaussian_cv, n_trials=100)

    except KeyboardInterrupt:
        pass

    # Best results with svm_spectrum_gaussian_cv