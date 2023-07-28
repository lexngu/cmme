import numpy as np


def trialtimefeature_sequence_as_multitrial_cell(input_sequence):
    """
    Converts an input sequence to the structure required by D-REX's estimate_suffstat.m
    :param input_sequence: np.array, trial x time x feature
    :return: trial x time x feature
    """
    if input_sequence.dtype == object:
        res = np.empty((input_sequence.shape[0],), dtype=object)
        for idx, e in enumerate(input_sequence):
            res[idx] = e.tolist()
        return res
    else:
        raise ValueError("input_sequence should be a np.array(dtype=object) of three dimensions.")


def trialtimefeature_sequence_as_singletrial_array(input_sequence):
    """
    Converts an input sequence to the array as used for D-REX's input sequences.
    :param input_sequence: np.array, trial x time x feature
    :return: time x feature
    """
    if input_sequence.dtype == object and input_sequence.shape[0] == 1:
        return np.array(input_sequence[0].tolist(), dtype=float)
    else:
        raise ValueError("input_sequence should be a np.array(dtype=object) of three dimensions, "
                         "and with a single trial.")


