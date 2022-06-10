import time
from util.mpiutil import rank0
import functools
import numpy as N
from numpy import linalg as LA

def myTiming_rank0(func):
    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        if rank0:
            print("Entering" + str(func))
            begin = time.perf_counter()
            print(begin)
        result = func(*args, **kwargs)
        if rank0:
            end = time.perf_counter()
            print(end)
            print("Elaspsed time", end - begin)
        return(result)
    return(decorated_function)

def myTiming(func):
    @functools.wraps(func)
    def decorated_function(*args, **kwargs):
        print("Entering" + str(func))
        begin = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print("Elaspsed time", end - begin)
        return(result)
    return(decorated_function)

def cache_last(func):
    """A simple decorator to cache the result of the last call to a function.
    """
    arg_cache = [None]
    kw_cache = [None]
    ret_cache = [None]

    @functools.wraps(func)
    def decorated(*args, **kwargs):

        if args != arg_cache[0] or kwargs != kw_cache[0]:
            # Generate cache value
            ret_cache[0] = func(*args, **kwargs)
            arg_cache[0] = args
            kw_cache[0] = kwargs
        # Fetch from cache
        return ret_cache[0]

    return decorated


def cache_last_n(n):
    def decorator(func):
        """A simple decorator to cache the result of the last call to a function.
        """
        arg_cache = [None] * n
        kw_cache = [None] * n
        ret_cache = [None] * n

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            aux = [args != arg_cache[i] or kwargs != kw_cache[i] for i in range(n)]
            if sum(aux) == n:
                ret_cache.insert(0, func(*args, **kwargs))
                arg_cache.insert(0, args)
                kw_cache.insert(0, kwargs)
                ret_cache.pop()
                arg_cache.pop()
                kw_cache.pop()
                result = ret_cache[0]
            else:
                result = ret_cache[aux.index(False)]
            # Fetch from cache
            return result
        return wrapper
    return decorator

def scaled_scalar(scaling):
    def decorator(func):
        def wrapper(xvec):
            if scaling:
                pvec = (N.exp(xvec) + N.exp(-xvec)) * .5 - 1
            else:
                pvec = xvec
            return func(pvec)
        return wrapper
    return decorator

def scaled_vector(scaling):
    def decorator(func):
        def wrapper(xvec):
            if scaling:
                result = func((N.exp(xvec) + N.exp(-xvec)) * .5 - 1) * (N.exp(xvec) - N.exp(-xvec))*.5
            else:
                result = func(xvec)
            return result
        return wrapper
    return decorator

def regularized_scalar(regularizing):
    def decorator(func):
        def wrapper(xvec):
            if regularizing:
                result = func(xvec) + LA.norm(xvec)
            else:
                result = func(xvec)
            return result
        return wrapper
    return decorator

def regularized_vector(regularizing):
    def decorator(func):
        def wrapper(xvec):
            if regularizing:
                result = func(xvec) + xvec/LA.norm(xvec)
            else:
                result = func(xvec)
            return result
        return wrapper
    return decorator

def cache_last_n_classfunc(func):
    arg_cache = [None]
    kw_cache = [None]
    ret_cache = [None]
    x_cache = [N.zeros(1)]

    @functools.wraps(func)
    def wrapper(self, x, *args, **kwargs):
        n = self.memorysize
        if arg_cache == [None]:
            arg_cache.extend([None]*(n-1))
            kw_cache.extend([None]*(n-1))
            ret_cache.extend([None] * (n - 1))
            x_cache.extend([N.zeros(1)] * (n - 1))
        aux = [args != arg_cache[i] or kwargs != kw_cache[i] or not N.allclose(x, x_cache[i]) for i in range(n)]
        if sum(aux) == n:
            ret_cache.insert(0, func(self, x, *args, **kwargs))
            arg_cache.insert(0, args)
            kw_cache.insert(0, kwargs)
            x_cache.insert(0, x)
            ret_cache.pop()
            arg_cache.pop()
            kw_cache.pop()
            x_cache.pop()
            result = ret_cache[0]
        else:
            result = ret_cache[aux.index(False)]
            # Fetch from cache
        return result
    return wrapper


def cache_last_n_specific(n):
    def decorator(func):
        """A simple decorator to cache the result of the last call to a function.
        """
        arg_cache = [None] * n
        kw_cache = [None] * n
        ret_cache = [None] * n
        x_cache = [N.zeros(1)] * n

        @functools.wraps(func)
        def wrapper(x, *args, **kwargs):
            aux = [args != arg_cache[i] or kwargs != kw_cache[i] or not N.allclose(x, x_cache[i]) for i in range(n)]
            if sum(aux) == n:
                ret_cache.insert(0, func(x, *args, **kwargs))
                arg_cache.insert(0, args)
                kw_cache.insert(0, kwargs)
                x_cache.insert(0, x)
                ret_cache.pop()
                arg_cache.pop()
                kw_cache.pop()
                x_cache.pop()
                result = ret_cache[0]
            else:
                result = ret_cache[aux.index(False)]
            # Fetch from cache
            return result

        return wrapper

    return decorator
