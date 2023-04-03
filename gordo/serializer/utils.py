from typing import get_origin, get_args, Union

try:
    from types import UnionType  # type: ignore
except ImportError:
    UnionType = None


def _is_exact_tuple_type(tp) -> bool:
    if tp is tuple:
        return True
    origin = get_origin(tp)
    return origin is tuple


def is_tuple_type(tp) -> bool:
    """
    Check if this type is a tuple.

    Examples
    --------
    >>> from typing import Optional, Tuple
    >>> is_tuple_type(tuple)
    True
    >>> is_tuple_type(Optional[tuple[int, int]])
    True
    >>> is_tuple_type(Tuple[str, str])
    True
    >>> is_tuple_type(list[str])
    False

    Parameters
    ----------
    tp
        Type for check.

    Returns
    -------
    """
    if _is_exact_tuple_type(tp):
        return True
    origin = get_origin(tp)
    if origin is Union or (UnionType is not None and origin is UnionType):
        args = get_args(tp)
        for arg in args:
            if not _is_exact_tuple_type(arg) and not (arg is type(None)):
                return False
        return True
    return False
