import numbers
from typing import Union

import numpy as np


def auto_convert_input_sequence(data: Union[list, np.ndarray]) -> np.ndarray:
    """
    Transform the data sequence to a unified representation of multi-trial, multi-feature data.
    That is, ensure that the data sequence is coded as np.array([trial1, trial2, ...], dtype=object), where
    each trial is a np.array(dtype=float) of shape (n_time, n_feature).

    * If data is a np.array(..., dtype=object), pass-through.
    * If data is a shallow list, then a single-trial, single-feature input sequence is assumed, i.e. the
    resulting numpy array has shape (1,), and the single contained trial has shape (time,1).
    * If data is a list of lists, then a multi-trial, single-feature input is assumed, i.e. the
    resulting numpy array has shape (trial,), and each contained trial has shape (time,1).
    * If data is a list of lists of lists, then a multi-trial, multi-feature input is assumed, i.e. the
    resulting numpy array has shape (trial,), and each contained trial has shape (time,feature).

    Parameters
    ----------
    data
        Arbitrary representation of an input sequence

    Returns
    -------
    np.ndarray
        Unified representation as numpy array
    """
    if type(data) is np.ndarray and data.dtype == object:
        # nothing to do, because already np.array(dtype=object)
        return data

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("input_sequence must not be empty!")

        firstlayer_firstelement = data[0]
        if isinstance(firstlayer_firstelement, numbers.Number):
            # then assume single trial and single feature
            return np.array([np.array([data], dtype=float).T], dtype=object)
        elif isinstance(firstlayer_firstelement, list):
            secondlayer_firstelement = firstlayer_firstelement[0]

            if isinstance(secondlayer_firstelement, numbers.Number):
                # multi-trial, but single feature
                data = [[trial] for trial in data]

            trials = list()
            for trial in data:
                trials.append(np.array(trial, dtype=float).T)
            return np.array(trials, dtype=object)  # trial x time x feature
    else:
        raise ValueError("input_sequence invalid! List expected.")
