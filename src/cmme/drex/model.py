import numbers
from abc import ABC
from pathlib import Path
from typing import Union

from .base import Prior
from .binding import DREXInstructionsFile
from .util import auto_convert_input_sequence
from ..lib.util import drex_default_results_file_path
import numpy as np

from ..lib.model import ModelBuilder


# TODO GMM has a paramater "beta", which is not a prior parameter, but a main function's parameter


class DREXInstructionBuilder(ModelBuilder, ABC):
    def __init__(self):
        # Use original default values
        super().__init__()
        self._input_sequence = None
        self._hazard = 0.01
        self._memory = np.inf
        self._maxhyp = np.inf
        # self._D = 1 # implicitly specified by prior
        self._change_decision_threshold = 0.01
        self._obsnz = 0
        self._predscale = 0.001

        self._prior = None

    def prior(self, prior: Prior):
        self._prior = prior
        return self

    def input_sequence(self, input_sequence: Union[list, np.ndarray]):
        """
        Sets the input sequence to process
        :param input_sequence: np.array of shape (time, feature)
        :return: self
        """
        iseq = auto_convert_input_sequence(input_sequence)
        [_, input_sequence_features] = iseq[0].shape

        # Check correspondence to prior (if present)
        if self._prior is not None:
            if self._prior.feature_count() < input_sequence_features:
                raise ValueError("input_sequence invalid! Its number of features must be equal the prior's.")

        # Automatically adjust obsnz if obsnz is scalar
        if isinstance(self._obsnz, numbers.Number):
            self._obsnz = [self._obsnz] * input_sequence_features

        self._input_sequence = iseq
        return self

    def hazard(self, hazard):
        """
        Sets the hazard rate(s)
        :param hazard: scalar, or np.array of shape (time,) matching the input sequence to process
        :return:
        """
        if type(hazard) is np.ndarray:
            # Check shape
            if len(hazard.shape) != 1:
                raise ValueError("Shape of hazard is invalid! Expected one dimension: time.")
            [hazard_times] = hazard.shape

            # Check correspondence to input sequence (if present)
            if len(hazard) > 1 and len(self._input_sequence) > 0:
                if len(self._input_sequence) != len(hazard_times):
                    raise ValueError("hazard invalid! There must be either one or as much as "
                                     "len(input_sequence) elements.")

            # Check value(s)
            if not ((hazard >= 0).all() and (hazard <= 1).all()):
                raise ValueError("hazard invalid! Value(s) must be within range of [0,1].")

        self._hazard = hazard
        return self

    def obsnz(self, obsnz: float):  # TODO per feature separate value?
        """
        Sets the observation noise which is the square root value of the number added to the (co)variance
        D-REX's calculations.

        :param obsnz: float
        :return:
        """
        self._obsnz = obsnz
        return self

    def memory(self, memory: int):
        """
        Sets the memory parameter which limits the number of previous hypotheses to process at each time step.
        :param memory: int or np.inf
        :return:
        """
        if not memory >= 2:
            raise ValueError("memory invalid! Value must be greater than or equal 2.")

        self._memory = memory
        return self

    def maxhyp(self, maxhyp: int):
        """
        Sets the maxhyp parameter which limits the number of hypotheses to keep at every time step.
        :param maxhyp: int or np.inf
        :return:
        """
        self._maxhyp = maxhyp
        return self

    def change_decision_threshold(self, change_decision_threshold):
        """
        Sets the change decision threshold.
        :param change_decision_threshold: float in range of [0,1].
        :return:
        """
        if not change_decision_threshold >= 0 and change_decision_threshold <= 1:
            raise ValueError("change_decision_threshold invalid! Value must be in range of [0,1].")
        self._change_decision_threshold = change_decision_threshold
        return self

    def build_instructions_file(self, results_file_path: Union[str, Path] = drex_default_results_file_path()) -> DREXInstructionsFile:
        return DREXInstructionsFile(str(results_file_path),
                                    self._input_sequence, self._prior,
                                    self._hazard, self._memory,
                                    self._maxhyp, self._obsnz,
                                    self._predscale, self._change_decision_threshold)

    def to_instructions_file(self) -> DREXInstructionsFile:
        return self.build_instructions_file()  # TODO remove build_instructions_file()

    def predscale(self, predscale: float):
        if not predscale > 0 and predscale <= 1:
            raise ValueError("predscale invalid! Value must be in range (0,1].")
        self._predscale = float(predscale)
        return self
