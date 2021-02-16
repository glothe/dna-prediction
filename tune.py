

import optuna

from load import *
from regression import *
from svm import *
from kernel import *

import sklearn.kernel_ridge
import sklearn.linear_model 
import sklearn.svm


# Xtr, Xva = split(load_X(1), .8)
Xtr, Xva = split(load_Xmat(1))
ytr, yva = split(load_y(1))


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
    # Best .6225 (idem krr ??)
    #      .600
    #      .7150
    #      

    # TODO Check convergence
    sig2 = trial.suggest_float("sig2", 1e-5, 1, log=True)
    C = trial.suggest_float("C", 1, 1e4, log=True)

    svm = SupportVectorMachine(kernel=gaussian_kernel(sig2), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)


def svm_spectrum(trial):
    C = trial.suggest_float("C", 1e-1, 1e2, log=True)
    m = trial.suggest_int("m", 3, 12)

    svm = SupportVectorMachine(kernel=spectrum_kernel(m), regularization=C)
    svm.fit(Xtr, ytr)
    return svm.accuracy(Xva, yva)

def random():
    n = len(yva)
    y_pred = np.sign(np.random.randn(n))
    err = y_pred - yva
    return np.mean(err * err)

study = optuna.create_study(direction="maximize")

try:
    study.optimize(sklearn_svm, n_trials=100)

except KeyboardInterrup:
    pass

from optuna.visualization import plot_contour
fig = plot_contour(study, params=["C", "sig2"])
fig.show()