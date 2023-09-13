from __future__ import annotations

from pathlib import Path
import os
from typing import Union

import numpy as np

from cmme.config import Config


def flatten_list(data, recursive=False) -> list:
    """
    Return a list with all elements of the input data. If data is not a list, a list with this data is returned.
    If data is a list, each contained element, which is a list itself, gets unpacked.
    Optionally, the input list is processed recursively.

    Parameters
    ----------
    data
        Input data
    recursive
        Whether to recursively unpack input list

    Returns
    -------
    list
        Result list
    """
    if data is None or data == []:
        return []
    if not isinstance(data, list):
        return [data]

    result = []
    for o in data:
        if isinstance(o, list):
            if recursive:
                result = [*result, *flatten_list(o)]
            else:
                result = [*result, *o]
        else:
            result.append(o)
    return result


def path_as_string_with_trailing_slash(path: Union[str, Path], expand_and_resolve=True) -> str:
    """
    Return a string representation of a Path object with trailing slash (or backslash, depending on the OS platform).

    Parameters
    ----------
    path
        Path
    expand_and_resolve
        Whether to try to resolve the path to the unique absolute path (including expanding "~" (user home path))

    Returns
    -------
    String representation
    """
    if path is None:
        raise ValueError("path must not be None!")
    if not isinstance(path, Path):
        path = Path(path)

    if expand_and_resolve:
        try:
            path = path.expanduser().resolve()
        except (RuntimeError, FileNotFoundError):
            pass  # Do nothing

    path_with_trailing_slash = os.path.join(path, '')

    return path_with_trailing_slash


def nparray_to_list(arr: np.ndarray) -> list:
    """
    Convert a (nested) numpy array to a Python list
    Parameters
    ----------
    arr: np.ndarray
        Input array to process
    Returns
    -------
    list
        Python list
    """
    result = []
    for e in arr:
        if type(e) is np.ndarray:
            result.append(nparray_to_list(e))
        else:
            result.append(e)
    return result
