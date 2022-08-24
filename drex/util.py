import numpy as np

def auto_convert_input_sequence(input_sequence):
    """
    Converts:
    1) np.array => pass-through
    2) flat list => input_sequence with 1 feature
    3) list of n flat lists => input_sequence with n features
    :param input_sequence: list or np.array
    :return: np.array with shape (time, feature)
    """
    if type(input_sequence) is np.array:
        return input_sequence
    elif isinstance(input_sequence, list):
        if len(input_sequence) < 1:
            return np.empty(shape=(0,0), dtype=float)
        if isinstance(input_sequence[0], list): # assume: [f1, f2, f3]
            return np.array(input_sequence, dtype=float).T # transpose (feature, time) => (time, feature)
        elif isinstance(input_sequence[0], int) or isinstance(input_sequence[0], float): # assume [x1, ...]
            return np.array([input_sequence], dtype=float).T # transpose (feature, time) => (time, feature)
    else:
        raise ValueError("input_sequence invalid!")