from pathlib import Path
from typing import Any

import matlab
import numpy.typing as npt
import numpy as np

from cmme.drex.distributions import UnprocessedDrexDistributionContainer, DistributionType, DrexDistributionContainer
from cmme.drex.util.matlab import MatlabWorker


class InstructionsFile:
    def __init__(self, instructions_file_path: Path, results_file_path: Path, input_sequence: npt.ArrayLike,
                 prior: DrexDistributionContainer, hazard: Any, memory: int, maxhyp: int, obsnz: float,
                 change_decision_threshold : float = None):
        # Checks
        input_sequence_length = len(input_sequence)
        if type(hazard) is int or type(hazard) is float:
            hazard_length = 1
        elif type(hazard) is np.ndarray:
            hazard_length = len(hazard)
        else:
            raise ValueError("hazard invalid! Should be scalar, or numpy.ndarray.")
        # If hazard is not scalar (i.e. more than one element), then there must be one value for each time step.
        if hazard_length > 1:
            if input_sequence_length != hazard_length:
                raise ValueError("Values invalid! If there is more than one hazard rate, the number of hazard rates must equal the number of input sequence data.")

        self.input_sequence = input_sequence
        """Input sequence: time, feature => 1"""
        self.instructions_file_path = instructions_file_path
        """Where to store the instructions file"""
        self.results_file_path = results_file_path
        """Where to store the model's results"""
        """Temporal dependence (for Gaussian, Lognormal, GMM) respectively interval size (Poisson)"""
        self.prior = prior
        """Prior distribution"""
        self.hazard = hazard
        """Hazard rate(s): scalar or list of numbers (for each input datum)"""
        self.obsnz = obsnz
        """Observation noise: feature => 1""" # TODO
        self.memory = memory
        """Number of most-recent hypotheses to calculate"""
        self.maxhyp = maxhyp
        """Number of hypotheses to calculate at every time step"""
        self.change_decision_threshold = change_decision_threshold
        """Threshold used for change detector"""

    def write_to_mat(self) -> Path:
        """
        Writes the instructions file to disk.
        :return: Path to instructions file
        """
        data = dict()

        # Add instructions for procesing an unprocessed prior using D-REX's estimate_suffstat.m
        if isinstance(self.prior, UnprocessedDrexDistributionContainer):
            data["estimate_suffstat"] = {
                "xs": matlab.double(self.prior._prior_input_sequence),
                "params": {
                    "distribution": self.prior._distribution.value,
                    "D": self.prior.D_value()
                }
            }
            if self.prior._distribution == DistributionType.GMM:
                data["estimate_suffstat"]["params"]["max_ncomp"] = self.prior._max_n_comp

        # Add instructions for invoking D-REX (run_DREX_model.m)
        data["run_DREX_model"] = {
            "x": matlab.double(self.input_sequence),
            "params": {
                "distribution": self.prior._distribution.value,
                "D": self.prior.D_value(),
                "hazard": matlab.double(self.hazard),
                "obsnz": matlab.double(self.obsnz),
                "memory": self.memory,
                "maxhyp": self.maxhyp
            },
        }
        if self.prior._distribution == DistributionType.GMM:
            data["run_DREX_model"]["params"]["max_ncomp"] = self.prior._max_n_comp

        # Add instructions for post_DREX_changedecision.m
        if self.change_decision_threshold != None:
            data["post_DREX_changedecision"] = {
                "threshold": self.change_decision_threshold
            }

        # Add results_file_path
        data["results_file_path"] = str(self.results_file_path)

        # Write and return
        return MatlabWorker.to_mat(data, str(self.instructions_file_path))
