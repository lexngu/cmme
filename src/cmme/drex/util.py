import numbers
from pathlib import Path

import numpy as np

from cmme.config import Config


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


def drex_default_instructions_file_path(alias: str = None, io_path: Path = Config().model_io_path()) -> str:
    instructions_file_filename = "drex-instructionsfile"
    instructions_file_filename = (instructions_file_filename + "-" + alias) if alias is not None \
        else instructions_file_filename
    instructions_file_filename = instructions_file_filename + ".mat"
    return str(io_path / instructions_file_filename)


def drex_default_results_file_path(alias: str = None, io_path: Path = Path(".")) -> str:
    results_file_filename = "drex-resultsfile"
    results_file_filename = (results_file_filename + "-" + alias) if alias is not None else results_file_filename
    results_file_filename = results_file_filename + ".mat"
    return str(io_path / results_file_filename)


def auto_convert_input_sequence(input_sequence):
    """
    Given any input, ensure that the input is coded as np.array([trial1, trial2, ...], dtype=object), where
    each trial is a np.array(dtype=float) of shape (n_time, n_feature).

    Converts:
    1) np.array(..., dtype=object) => pass-through
    2) [1, 2, 3, ...] => input_sequence with 1 trial and 1 feature
    3) [[1, 2, 3, ...], [1, 2, 3, ...], ...] => input_sequence with 1 trial and n features
    4) [[[1, 2, 3, ...], [1, 2, 3, ...], ...], [...], ...] => input_sequence with n trials and m features
    :param input_sequence: list or np.array
    :return: np.array with shape (trial,)
    """
    if type(input_sequence) is np.ndarray and input_sequence.dtype == object:
        # nothing to do, because already np.array(dtype=object)
        return input_sequence

    if isinstance(input_sequence, list):
        if len(input_sequence) == 0:
            raise ValueError("input_sequence must not be empty!")

        firstlayer_firstelement = input_sequence[0]
        if isinstance(firstlayer_firstelement, numbers.Number):
            # then assume single trial and single feature
            return np.array([np.array([input_sequence], dtype=float).T], dtype=object)
        elif isinstance(firstlayer_firstelement, list):
            secondlayer_firstelement = firstlayer_firstelement[0]

            if isinstance(secondlayer_firstelement, numbers.Number):
                # multi-trial, but single feature
                input_sequence = [[trial] for trial in input_sequence]

            trials = list()
            for trial in input_sequence:
                trials.append(np.array(trial, dtype=float).T)
            return np.array(trials, dtype=object)  # trial x time x feature
    else:
        raise ValueError("input_sequence invalid! List expected.")
