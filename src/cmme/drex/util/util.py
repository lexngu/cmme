from datetime import datetime

import numpy as np

from cmme.config import Config


def auto_convert_input_sequence(input_sequence):
    """
    Converts:
    1) np.ndarray => pass-through
    2) flat list => input_sequence with 1 feature
    3) list of n flat lists => input_sequence with n features
    :param input_sequence: list or np.array
    :return: np.array with shape (time, feature)
    """
    if type(input_sequence) is np.ndarray:
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

def transform_multifeature_singletrial_input_sequence_for_estimate_suffstat(input_sequence):
    """
    Converts an input sequence to the structure required by D-REX's estimate_suffstat.m (using double arrays, since cell arrays are not supported in Python)
    :param input_sequence: np.array, time x feature
    :return: time x 1 x feature
    """
    res = input_sequence[..., np.newaxis] # introduce third dimension
    res = np.transpose(res, (0, 2, 1)) # format as: time x trial (=1) x feature
    return res

def drex_default_instructions_file_path(alias = None):
    instructions_file_filename = datetime.now().isoformat().replace("-", "").replace(":", "").replace(".", "")
    instructions_file_filename = instructions_file_filename + "-drex-instructionsfile"
    instructions_file_filename = instructions_file_filename + "-" + alias if alias is not None else instructions_file_filename
    instructions_file_filename = instructions_file_filename + ".mat"
    return Config().model_io_path() / instructions_file_filename

def drex_default_results_file_path(alias = None):
    results_file_filename = datetime.now().isoformat().replace("-", "").replace(":", "").replace(".", "")
    results_file_filename = results_file_filename + "-drex-resultsfile"
    results_file_filename = results_file_filename + "-" + alias if alias is not None else results_file_filename
    results_file_filename = results_file_filename + ".mat"
    return Config().model_io_path() / results_file_filename