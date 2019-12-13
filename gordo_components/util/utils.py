import functools
import inspect
from typing import Callable


def capture_args(method: Callable):
    """
    Decorator to capture ``args`` and ``kwargs`` passed to a given method
    Assumed there is a ``self`` to which it assigned the attribute to as ``_params``
    as one dict of kwargs

    Parameters
    ----------
    method: Callable
        Some method of an object, with 'self' as the first parameter

    Returns
    -------
    Any
        Returns whatever the original method would return
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        """
        Need to map args to the parameters in the method and then update that
        param dict with explicitly provided kwargs and assign that to _params
        """
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
