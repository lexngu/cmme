from .distribution import Distribution
from .prior import Prior


class InstructionsFile:
    def __init__(self, distribution: Distribution, D, prior: Prior, hazard, obsnz, memory, maxhyp, change_decision_threshold, input_sequence, results_file_path):
        input_sequence_length = len(input_sequence)
        hazard_length = len(hazard)
        # If hazard is not scalar (i.e. more than one element), then there must be one value for each time step.
        if hazard_length > 1:
            if input_sequence_length != hazard_length:
                raise ValueError("Values invalid! If there is more than one hazard rate, the number of hazard rates must equal the number of input sequence data.")

        self.distribution = distribution
        """Distribution type for all further calculations"""
        self.D = D
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
        self.input_sequence = input_sequence
        """Input sequence: time, feature => 1"""
        self.results_file_path = results_file_path
        """Where to store the model's results"""

class InstructionsFileBuilder:
    """Helps creating InstructionsFile instances. Uses the default values from the original implementation."""
    def __init__(self):
        pass