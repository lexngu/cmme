from __future__ import annotations

import numbers
from abc import ABC
from typing import Union, List

from .base import Prior
from .binding import DREXInstructionsFile, DREXResultsFile
from .util import transform_to_unified_drex_input_sequence_representation
import numpy as np

from .worker import MatlabWorker
from ..lib.model import ModelBuilder, Model


class DREXInstructionBuilder(ModelBuilder, ABC):
    def __init__(self):
        """
        D-REX builder. Uses the default values, i.e.:

        * hazard = 0.01
        * memory = inf
        * maxhyp = inf
        * obsnz = 0
        * max_ncomp = 10 (relevant for GMM priors)
        * beta = 0.001 (relevant for GMM priors)
        * predscale = 0.01
        * change decision threshold = 0.01

        Further required values:

        * prior
        * input sequence

        """
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
        self._max_ncomp = 10
        self._beta = 0.001

        self._prior = None

    def prior(self, prior: Prior) -> DREXInstructionBuilder:
        """
        Set the (un)processed prior.

        Parameters
        ----------
        prior
            Prior
        Returns
        -------
        DREXInstructionBuilder
            self
        """
        self._prior = prior
        return self

    def input_sequence(self, input_sequence: Union[list, np.ndarray]) -> DREXInstructionBuilder:
        """
        Set the input sequence

        Parameters
        ----------
        input_sequence
            np.ndarray of shape (time, feature)

        Returns
        -------
        DREXInstructionBuilder
            self
        """
        iseq = transform_to_unified_drex_input_sequence_representation(input_sequence)
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

    def hazard(self, hazard: Union[float, List[float]]) -> DREXInstructionBuilder:
        """
        Set the hazard rate(s)

        Parameters
        ----------
        hazard
            Value between [0,1], or list of such values (as much as the input sequence has elements)

        Returns
        -------
        DREXInstructionBuilder
            self
        """
        if isinstance(hazard, list):
            hazard = np.array(hazard)

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

    def obsnz(self, obsnz: Union[float, List[float]]) -> DREXInstructionBuilder:  # TODO per feature separate value
        """
        Set the observation noise

        Parameters
        ----------
        obsnz
            Single values used across features, or list of values (as much as the input sequence has features)

        Returns
        -------
        DREXInstructionBuilder
            self
        """
        self._obsnz = obsnz
        return self

    def memory(self, memory: int) -> DREXInstructionBuilder:
        """
        Set the memory parameter which limits the number of previous hypotheses to process at each time step.
        Parameters
        ----------
        memory
            positive int (greather than or equal 2) or np.inf
        Returns
        -------
        DREXInstructionBuilder
            self
        """
        if not memory >= 2:
            raise ValueError("memory invalid! Value must be greater than or equal 2.")

        self._memory = memory
        return self

    def maxhyp(self, maxhyp: int) -> DREXInstructionBuilder:
        """
        Set the maxhyp parameter which limits the number of previous hypotheses to process at each time step.

        Parameters
        ----------
        maxhyp
            positive int or np.inf
        Returns
        -------
        DREXInstructionBuilder
            self
        """
        self._maxhyp = maxhyp
        return self

    def change_decision_threshold(self, change_decision_threshold: float) -> DREXInstructionBuilder:
        """
        Set the change decision threshold.

        Parameters
        ----------
        change_decision_threshold
            float in range [0,1]

        Returns
        -------
        DREXInstructionBuilder
            self
        """
        if not change_decision_threshold >= 0 and change_decision_threshold <= 1:
            raise ValueError("change_decision_threshold invalid! Value must be in range of [0,1].")
        self._change_decision_threshold = change_decision_threshold
        return self

    def predscale(self, predscale: float) -> DREXInstructionBuilder:
        """
        Set the predscale value.

        Parameters
        ----------
        predscale
            float in range [0,1]

        Returns
        -------
        DREXInstructionBuilder
            self
        """
        if not predscale > 0 and predscale <= 1:
            raise ValueError("predscale invalid! Value must be in range (0,1].")
        self._predscale = float(predscale)
        return self

    def max_ncomp(self, max_ncomp: int) -> DREXInstructionBuilder:
        """
        Set the max_ncomp value D-REX's handling of GMM priors.

        Parameters
        ----------
        max_ncomp
            Maximum number of components

        Returns
        -------
        DREXInstructionBuilder
            self
        """
        if not (max_ncomp > 0 and isinstance(max_ncomp, int)):
            raise ValueError("max_ncomp invalid! Value must be positive and integer.")

        self._max_ncomp = max_ncomp

        return self

    def beta(self, beta: float) -> DREXInstructionBuilder:
        """
        Set the beta value for D-REX's handling of GMM priors.

        Parameters
        ----------
        beta
            Threshold for new GMM components.

        Returns
        -------
        DREXInstructionBuilder
            self
        """
        if not (0 <= beta <= 1):
            raise ValueError("beta invalid! Value must be between 0 and 1.")

        self._beta = beta

        return self

    def to_instructions_file(self) -> DREXInstructionsFile:
        return DREXInstructionsFile(self._input_sequence, self._prior,
                                    self._hazard, self._memory,
                                    self._maxhyp, self._obsnz,
                                    self._max_ncomp, self._beta,
                                    self._predscale, self._change_decision_threshold)


class DREXModel(Model):
    """
    High-level interface for using D-REX.
    Using +instance+, one can hyper-parameterize D-REX.
    """

    def __init__(self):
        super().__init__()

    def run(self, instructions_file_path) -> DREXResultsFile:
        results_file_path = MatlabWorker.run_model(instructions_file_path)
        results_file = DREXResultsFile.load(results_file_path)
        return results_file

    @staticmethod
    def run_instructions_file_at_path(file_path: str) -> DREXResultsFile:
        return DREXModel().run(file_path)
