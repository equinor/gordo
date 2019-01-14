import os
from contextlib import contextmanager


@contextmanager
def temp_env_vars(**kwargs):
    """
    Temporarily set the process environment variables
    """
    _env = os.environ.copy()

    for key in kwargs:
        os.environ[key] = kwargs[key]

    yield

    os.environ.clear()
    os.environ.update(_env)
