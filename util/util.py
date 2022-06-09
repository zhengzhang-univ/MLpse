import time
from util.mpiutil import rank0
import functools

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