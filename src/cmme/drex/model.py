from pathlib import Path
import cmme.drex.util.matlab
from .distribution.prior import Prior
from .instance import DREXInstance
from .instructions_file import InstructionsFile
from .results_file import ResultsFile
from .util.matlab import MatlabWorker
from .util.util import *


class DREXModel:
    """
    High-level interface for using D-REX.
    Using +instance+, one can hyper-parameterize D-REX.
    """

    def __init__(self, instance: DREXInstance):
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
        results_file = cmme.drex.results_file.parse_results_file(results['results_file_path'])
        return results_file
