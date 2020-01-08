import functools
import inspect
from typing import Callable


def capture_args(method: Callable):
    """
    Decorator that captures args and kwargs passed to a given method.
    This assumes the decorated method has a self, which has a dict of
    kwargs assigned as an attribute named _params.

    Parameters
    ----------
    method: Callable
        Some method of an object, with 'self' as the first parameter.

    Returns
    -------
    Any
        Returns whatever the original method would return
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):

        sig_params = inspect.signature(method).parameters.items()

        # Get the default values for the method signature
        params = {
            param: value.default
            for param, value in sig_params
            if value.default is not inspect.Parameter.empty and param != "self"
        }

        # Update with arg values provided
        arg_map = dict()
        for arg_val, arg_key in zip(
            args, (arg for arg in inspect.getfullargspec(method).args if arg != "self")
        ):
            arg_map[arg_key] = arg_val

        # Update params with args/kwargs provided in the current call
        params.update(arg_map)
        params.update(kwargs)

        self._params = params
        return method(self, *args, **kwargs)

    return wrapper
