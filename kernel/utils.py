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
