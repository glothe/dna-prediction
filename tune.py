

import optuna

from load import *
from regression import *
from svm import *
from kernel import *

import sklearn.kernel_ridge
import sklearn.linear_model 
import sklearn.svm

## Parameters
index = 2
split_param = .8
# Xtr, Xva = split(load_X(index), split_param)
Xtr, Xva = split(load_Xmat(index), split_param)
ytr, yva = split(load_y(index), split_param)

## All test functions
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

def krr_gaussian(trial):
    # Best .62 (pq mieux que sklearn ???)
    sig2 = trial.suggest_float("sig2", 1e-5, 1e-2, log=True)
    lambda_ = trial.suggest_float("lambda_", 1, 1e2, log=True)

    klr = KernelRidgeRegression(kernel=gaussian_kernel(sig2), regularization=lambda_)
    klr.fit(Xtr, ytr)

    return klr.accuracy(Xva, yva)

def klr_gaussian(trial):
    # Best 
    #   .62 (idem krr ??)
    sig2 = trial.suggest_float("sig2", 1e-5, 1, log=True)
    lambda_ = trial.suggest_float("lambda_", 1, 1e2, log=True)

    klr = KernelLogisticRegression(kernel=gaussian_kernel(sig2), regularization=lambda_)
    klr.fit(Xtr, ytr)

    return klr.accuracy(Xva, yva)

def svm_gaussian(trial):
    # Best  0.62 (index=0)      {'sig2': 0.0004475252677121083, 'C': 3.7371130579973366}
    #       0.6225 (index=1)    {'sig2': 0.01877452515739, 'C': 1.7176296305911665}
    #       0.7275 (index=2)    {'sig2': 0.006672040107687773, 'C': 1.0330032957410673}

    # TODO Check convergence
    sig2 = trial.suggest_float("sig2", 1e-5, 1e-1, log=True)
    C = trial.suggest_float("C", 1e-2, 10, log=True)

    svm = SupportVectorMachine(kernel=gaussian_kernel(sig2), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)

def svm_spectrum(trial):
    # Best  0.6300 (index=0)      {'C': 0.2681280835940456, 'm': 4}
    #       0.6175 (index=1)    {'C': 56.883446994457, 'm': 4} #with value: 0.615 and parameters: {'C': 16.685807660880197, 'm': 4}
    #       0.6975 (index=2)    {'C': 0.24075076252842445, 'm': 4}
    C = trial.suggest_float("C", 1e-1, 1e2, log=True)
    m = trial.suggest_int("m", 3, 5)

    svm = SupportVectorMachine(kernel=spectrum_kernel(m), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)

def svm_substring(trial):
    C = trial.suggest_float("C", 1e-1, 1e2, log=True)
    p = 2 #trial.suggest_int("p", 2, 2)
    lam = trial.suggest_float("lam", 0, 1)

    svm = SupportVectorMachine(kernel=substring_kernel(p, lam), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)

def random():
    n = len(yva)
    y_pred = np.sign(np.random.randn(n))
    err = y_pred - yva
    return np.mean(err * err)

print("_______ Dataset index=", index)
study = optuna.create_study(direction="maximize")

try:
    study.optimize(svm_gaussian, n_trials=100)

except KeyboardInterrupt:
    pass

# from optuna.visualization import plot_contour
# fig = plot_contour(study, params=["C", "sig2"])
# fig.show()