import numpy as np
from .distribution import Distribution
from .prior import Prior
from .util import auto_convert_input_sequence


class DREXInstance:
    def __init__(self, distribution: Distribution):
        self._distribution: Distribution = distribution
        self._input_sequence = auto_convert_input_sequence([])
        self._hazard = np.array([0.01], dtype=float)
        self._memory = np.inf
        self._maxhyp = np.inf
        self._D = 1
        self._prior: Prior = None
        self._change_decision_threshold = 0.01
        self._observation_noise = 0.0

    def with_input_sequence(self, input_sequence):
        """
        Sets the input sequence to process
        :param input_sequence: np.array of shape (time, feature)
        :return: self
        """
        iseq = auto_convert_input_sequence(input_sequence)

        # Check shape
        if len(iseq.shape) != 2:
            raise ValueError("Shape of input_sequence invalid! Expected two dimensions: time, feature.")
        [input_sequence_times, input_sequence_features] = iseq.shape

        # Check correspondence to prior (if present)
        if self._prior is not None:
            if self._prior.features_count() < input_sequence_features:
                raise ValueError("input_sequence invalid! Its number of features must be equal the prior's.")

        self._input_sequence = iseq

    def with_hazard(self, hazard):
        """
        Sets the hazard rate(s)
        :param hazard: scalar or np.array of shape (time,) matching the input sequence to process
        :return:
        """
        if type(hazard) is int or type(hazard) is float:
            hazard = np.array(hazard, dtype=float)
        if type(hazard) is np.ndarray:
            # Check shape
            if len(hazard.shape) != 1:
                raise ValueError("Shape of hazard is invalid! Expected one dimension: time.")
            [hazard_times] = hazard.shape

            # Check correspondence to input sequence (if present)
            if len(hazard) > 1 and len(self._input_sequence) > 0:
                if len(self._input_sequence) != len(hazard_times):
                    raise ValueError("hazard invalid! There must be either one or as much as len(input_sequence) elements.")

            # Check value(s)
            if not ((hazard >= 0).all() and (hazard <= 1).all()):
                raise ValueError("hazard invalid! Value(s) must be within range of [0,1].")
        else:
            raise ValueError("hazard invalid! Expected type: scalar or np.array.")

        self._hazard = hazard

    def with_observation_noise(self, obsnz: float):
        """
        Sets the observation noise which is the square root value of the number added to the (co)variance D-REX's calculations.
        :param obsnz: float
        :return:
        """
        self._observation_noise = obsnz

    def with_memory(self, memory: int):
        """
        Sets the memory parameter which limits the number of previous hypotheses to process at each time step.
        :param memory: int or np.inf
        :return:
        """
        if not memory >= 2:
            raise ValueError("memory invalid! Value must be greater than or equal 2.")
        if not memory >= self._maxhyp:
            raise ValueError("memory invalid! Value must be greater than or equal maxhyp.")

        self._memory = memory

    def with_maxhyp(self, maxhyp: int):
        """
        Sets the maxhyp parameter which limits the number of hypotheses to keep at every time step.
        :param maxhyp: int or np.inf
        :return:
        """
        if not maxhyp <= self._memory:
            raise ValueError("maxhyp invalid! Value must be less than or equal memory.")
        self._maxhyp = maxhyp

    def with_D(self, D: int):
        """
        Sets the value of the parameter D.
        :param D: positive int
        :return:
        """
        # Check value
        if not D >= 1:
            raise ValueError("D invalid! Value must be greater than or equal 1.")
        # Check correspondence to prior (if present)
        if self._prior is not None:
            if self._prior.D() != D:
                raise ValueError("D invalid! Value must match prior's D value.")
        self._D = D

    def with_prior(self, prior: Prior):
        """
        Sets the prior.
        :param prior:
        :return:
        """
        # Check correspondence to input_sequence (if present)
        if len(self._input_sequence) > 0:
            [input_sequence_times, input_sequence_features] = self._input_sequence.shape
            if prior.features_count() < input_sequence_features:
                raise ValueError("input_sequence invalid! Its number of features must be equal the prior's.")
        # Check correspondence to distribution
        if prior.distribution() != self._distribution:
            raise ValueError("prior invalid! Its distribution-value must be equal to _distribution.")

        self._prior = prior

    def with_change_decision_threshold(self, change_decision_threshold):
        """
        Sets the change decision threshold.
        :param threshold: float in range of [0,1].
        :return:
        """
        if not change_decision_threshold >= 0 and change_decision_threshold <= 1:
            raise ValueError("change_decision_threshold invalid! Value must be in range of [0,1].")
        self._change_decision_threshold = change_decision_threshold