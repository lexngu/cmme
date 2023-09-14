from __future__ import annotations

import numbers
from typing import Union

import numpy as np


def transform_to_unified_drex_input_sequence_representation(data: Union[list, np.ndarray]) -> np.ndarray:
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

    if isinstance(data, list) or isinstance(data, np.ndarray):
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


freq_to_midi = {
    8.18: 0, 8.66: 1, 9.18: 2, 9.72: 3, 10.3: 4, 10.91: 5, 11.56: 6, 12.25: 7, 12.98: 8, 13.75: 9, 14.57: 10, 15.43: 11, 16.35: 12, 17.32: 13, 18.35: 14, 19.45: 15, 20.6: 16, 21.83: 17, 23.12: 18, 24.5: 19, 25.96: 20, 27.5: 21, 29.14: 22, 30.87: 23, 32.7: 24, 34.65: 25, 36.71: 26, 38.89: 27, 41.2: 28, 43.65: 29, 46.25: 30, 49.0: 31, 51.91: 32, 55.0: 33, 58.27: 34, 61.74: 35, 65.41: 36, 69.3: 37, 73.42: 38, 77.78: 39, 82.41: 40, 87.31: 41, 92.5: 42, 98.0: 43, 103.83: 44, 110.0: 45, 116.54: 46, 123.47: 47, 130.81: 48, 138.59: 49, 146.83: 50, 155.56: 51, 164.81: 52, 174.61: 53, 185.0: 54, 196.0: 55, 207.65: 56, 220.0: 57, 233.08: 58, 246.94: 59, 261.63: 60, 277.18: 61, 293.66: 62, 311.13: 63, 329.63: 64, 349.23: 65, 369.99: 66, 392.0: 67, 415.3: 68, 440.0: 69, 466.16: 70, 493.88: 71, 523.25: 72, 554.37: 73, 587.33: 74, 622.25: 75, 659.26: 76, 698.46: 77, 739.99: 78, 783.99: 79, 830.61: 80, 880.0: 81, 932.33: 82, 987.77: 83, 1046.5: 84, 1108.73: 85, 1174.66: 86, 1244.51: 87, 1318.51: 88, 1396.91: 89, 1479.98: 90, 1567.98: 91, 1661.22: 92, 1760.0: 93, 1864.66: 94, 1975.53: 95, 2093.0: 96, 2217.46: 97, 2349.32: 98, 2489.02: 99, 2637.02: 100, 2793.83: 101, 2959.96: 102, 3135.96: 103, 3322.44: 104, 3520.0: 105, 3729.31: 106, 3951.07: 107, 4186.01: 108, 4434.92: 109, 4698.64: 110, 4978.03: 111, 5274.04: 112, 5587.65: 113, 5919.91: 114, 6271.93: 115, 6644.88: 116, 7040.0: 117, 7458.62: 118, 7902.13: 119, 8372.02: 120, 8869.84: 121, 9397.27: 122, 9956.06: 123, 10548.08: 124, 11175.3: 125, 11839.82: 126, 12543.85: 127
}


def midi_to_equal_tempered_fundamental_frequency(midi_note_number: int, standard_concert_A_pitch: int = 440, precision_dp: int = 2) -> float:
    return round(2**((midi_note_number-69)/12)*standard_concert_A_pitch, precision_dp)


def equal_tempered_fundamental_frequency_to_midi(fundamental_frequency: float) -> int:
    freq = round(fundamental_frequency, 2)
    if freq in freq_to_midi:
        return freq_to_midi[freq]
    else:
        raise ValueError("Invalid frequency! Could not convert to midi note number.")
