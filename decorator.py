import functools
from time import time


def log_time(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t = time()
            result = func(*args, **kwargs)
            logger.info(f'{func.__name__} completed in {time() - t} seconds')
            return result

        return wrapper
    return decorator
