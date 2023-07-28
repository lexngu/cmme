from ctypes import Union

import numpy as np


def as_multifeature_timeseries(data: Union[list, np.ndarray]) -> np.ndarray:
    """
    Converts an arbitrarily represented "input sequence" into a unified representation as numpy array.
    The output has the shape (time, feature).

    If the input already is a numpy array with dimension=2, nothing happens.

    Parameters
    ----------
    data
        Arbitrary representation of an input sequence

    Returns
    -------
    A representation of the input data as numpy array with shape (time, feature).
    """
    raise NotImplementedError