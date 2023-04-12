import numbers
from pathlib import Path
from .base import Prior
from .binding import MatlabWorker, InstructionsFile, ResultsFile, parse_results_file
from .util import auto_convert_input_sequence, drex_default_instructions_file_path, drex_default_results_file_path
import numpy as np

class DREXInstructionBuilder:
    def __init__(self):
        # Use original default values
        self._input_sequence = auto_convert_input_sequence([])
        self._hazard = 0.01
        self._memory = np.inf
        self._maxhyp = np.inf
        # self._D = 1 # implicitly specified by prior
        self._change_decision_threshold = 0.01
        self._obsnz = 0

        self._prior = None

    def prior(self, prior: Prior):
        self._prior = prior
        return self

    def input_sequence(self, input_sequence):
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
                    raise ValueError("hazard invalid! There must be either one or as much as len(input_sequence) elements.")

            # Check value(s)
            if not ((hazard >= 0).all() and (hazard <= 1).all()):
                raise ValueError("hazard invalid! Value(s) must be within range of [0,1].")

        self._hazard = hazard
        return self

    def obsnz(self, obsnz: float): # TODO per feature separate value?
        """
        Sets the observation noise which is the square root value of the number added to the (co)variance D-REX's calculations.
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
        :param threshold: float in range of [0,1].
        :return:
        """
        if not change_decision_threshold >= 0 and change_decision_threshold <= 1:
            raise ValueError("change_decision_threshold invalid! Value must be in range of [0,1].")
        self._change_decision_threshold = change_decision_threshold
        return self

class DREXModel:
    """
    High-level interface for using D-REX.
    Using +instance+, one can hyper-parameterize D-REX.
    """

    def __init__(self, instance: DREXInstructionBuilder):
        """Creates a D-REX instance with D-REX's current default values"""
        self.instance = instance

    def to_instructions_file(self, instructions_file_path = drex_default_instructions_file_path(), results_file_path = drex_default_results_file_path()) -> InstructionsFile:
        """
        Returns an instruction file object. To write it to disk, use its write_instructions_file() method
        :param instructions_file_path: path to where the instructions file should be stored
        :param results_file_path:  path to where the results file should be stored
        :return: instructions file object
        """
        return InstructionsFile(instructions_file_path, results_file_path,
                                self.instance._input_sequence, self.instance._prior, self.instance._hazard, self.instance._memory, self.instance._maxhyp, self.instance._obsnz,
                                self.instance._change_decision_threshold)

    def run(self, instructions_file_path : Path = drex_default_instructions_file_path(), results_file_path : Path = drex_default_results_file_path()) -> ResultsFile:
        instructions_file = self.to_instructions_file(instructions_file_path, results_file_path)
        instructions_file.write_to_mat()
        results = MatlabWorker.run_model(instructions_file_path)
        results_file = parse_results_file(results['results_file_path'])
        return results_file
