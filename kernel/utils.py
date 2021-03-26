"""
Memoization function to avoid repeting similar computations
+ Normalization function
"""
import numpy as np


def memoize_id(func):
    """ 
    Memoize the function by object id. 
    Useful during tuning.
    Use with caution !!
    """
    # memoize by object id
    func.cache = {}

    def wrapper(*args):
        id_ = tuple([id(arg) for arg in args])
        if id_ not in func.cache:
            func.cache[id_] = func(*args)
        return func.cache[id_]

    return wrapper


def normalize_kernel(K, rows: np.ndarray=None, columns: np.ndarray=None):
    if rows is None or columns is None:
        # Assumes K is symmetric
        rows = columns = 1 / np.sqrt(np.diagonal(K)) 
    else:
        rows = 1 / np.sqrt(rows)
        columns = 1 / np.sqrt(columns)

    return K * rows[:, None] * columns[None,]