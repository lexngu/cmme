from abc import ABC
import random

from cmme.ppmdecay.base import EscapeMethod, ModelType
from cmme.ppmdecay.binding import InstructionsFile, PPMSimpleInstructionsFile, PPMDecayInstructionsFile, \
    ResultsMetaFile, invoke_model, parse_results_meta_file
from cmme.ppmdecay.util import ppmdecay_default_results_file_path, ppmdecay_default_instructions_file_path


class PPMInstance(ABC):
    def __init__(self, model_type: ModelType):
        self._model_type = model_type

        self._order_bound = 10
        self._alphabet_levels = []
        self._input_sequence = []

    def order_bound(self, order_bound):
        """

        :param order_bound: non-negative int
        :return:
        """
        if not order_bound > 0:
            raise ValueError("order_bound invalid! Value must be greater than or equal 0.")
        self._order_bound = order_bound
        return self

    def alphabet_levels(self, alphabet_levels):
        """

        :param alphabet_levels:
        :return: list
        """
        # Check length
        if not len(alphabet_levels) >= 1:
            raise ValueError("alphabet_levels invalid! It must contain at least one element.")

        self._alphabet_levels = alphabet_levels
        return self

    def input_sequence(self, input_sequence, input_time_sequence = None):
        """

        :param input_sequence:
        :param input_time_sequence: Relevant for PPMDecayInstance. If None, a default time sequence is generated: [1, 2, 3, ...]
        :return:
        """
        if not len(input_sequence) > 1:
            raise ValueError("input_sequence invalid! Length must be greater than or equal 1.")
        # check correspondence with alphabet_levels
        if len(self._alphabet_levels) >= 1:
            if not set(input_sequence).issubset(set(self._alphabet_levels)):
                raise ValueError("input_sequence invalid! Its elements must be compatible to alphabet_levels.")

        # auto-generate time sequence if None
        its = input_time_sequence
        if its == None:
            its = list(range(len(input_sequence)))
        # check correspondence of both sequences
        if len(input_sequence) != len(its):
            raise ValueError("input_sequence and input_time_sequence invalid! Length must match.")

        # Set attributes
        self._input_sequence = input_sequence
        self._input_time_sequence = its

        return self

class PPMSimpleInstance(PPMInstance):
    def __init__(self):
        super().__init__(ModelType.SIMPLE)
        self._shortest_deterministic = True
        self._exclusion = True
        self._update_exclusion = True
        self._escape_method = EscapeMethod.C

    def shortest_deterministic(self, shortest_deterministic):
        self._shortest_deterministic = shortest_deterministic
        return self

    def exclusion(self, exclusion):
        self._exclusion = exclusion
        return self

    def update_exclusion(self, update_exclusion):
        self._update_exclusion = update_exclusion
        return self

    def escape_method(self, escape_method: EscapeMethod):
        self._escape_method = escape_method
        return self


class PPMDecayInstance(PPMInstance):
    def __init__(self):
        super().__init__(ModelType.DECAY)

        self._buffer_weight = 1
        self._buffer_length_time = 0
        self._buffer_length_items = 0
        self._only_learn_from_buffer = False
        self._only_predict_from_buffer = False

        self._stm_weight = 1
        self._stm_duration = 0

        self._ltm_weight = 1
        self._ltm_half_life = 10
        self._ltm_asymptote = 0
        self._noise = 0

        self._seed = random.randint(1, pow(2, 31)-1)

    def input_time_sequence(self, input_time_sequence):
        self._input_time_sequence = input_time_sequence
        return self

    def buffer_weight(self, buffer_weight):
        if not (buffer_weight >= self._stm_weight and buffer_weight >= self._ltm_weight):
            raise ValueError("buffer_weight invalid! Value must be greater than or equal both stm_weight and ltm_weight.")
        self._buffer_weight = buffer_weight
        return self

    def buffer_length_time(self, buffer_length_time):
        if not buffer_length_time > 0:
            raise ValueError("buffer_length_time invalid! Value must be greater than or equal 0.")
        self._buffer_length_time = buffer_length_time
        return self

    def buffer_length_items(self, buffer_length_items):
        if not buffer_length_items > 0:
            raise ValueError("buffer_length_items invalid! Value must be greater than or equal 0.")

        self._buffer_length_items = buffer_length_items
        return self

    def only_learn_from_buffer(self, only_learn_from_buffer):
        self._only_learn_from_buffer = only_learn_from_buffer
        return self

    def only_predict_from_buffer(self, only_predict_from_buffer):
        self._only_predict_from_buffer = only_predict_from_buffer
        return self

    def stm_weight(self, stm_weight):
        if not (self._buffer_weight >= stm_weight >= self._ltm_weight):
            raise ValueError("stm_weight invalid! Value must be less than or equal buffer_weight and greater than or equal ltm_weight.")
        self._stm_weight = stm_weight
        return self

    def stm_duration(self, stm_duration):
        if not stm_duration > 0:
            raise ValueError("stm_duration invalid! Value must be greater than or equal 0.")
        self._stm_duration = stm_duration
        return self

    def ltm_weight(self, ltm_weight):
        if not (self._buffer_weight >= ltm_weight and self._stm_weight >= ltm_weight):
            raise ValueError("ltm_weight invalid! Value must be less than or equal both buffer_weight and stm_weight.")
        self._ltm_weight = ltm_weight
        return self

    def ltm_half_life(self, ltm_half_life):
        self._ltm_half_life = ltm_half_life
        return self

    def ltm_asymptote(self, ltm_asymptote):
        if not (ltm_asymptote <= self._ltm_weight):
            raise ValueError("ltm_asymptote invalid! Value must be less than or equal ltm_weight.")
        self._ltm_asymptote = ltm_asymptote
        return self

    def noise(self, noise: float):
        """

        :param noise: Variance of the Gaussian noise.
        :return:
        """
        self._noise = noise
        return self

    def seed(self, seed: int):
        """

        :param seed: Seed used by PPM-Decay's random number generator.
        :return:
        """
        self._seed = seed
        return self

class PPMModel:
    def __init__(self, instance: PPMInstance):
        self.instance = instance
        self.model_type = instance._model_type

    def to_instructions_file(self, results_file_path = ppmdecay_default_results_file_path()) -> InstructionsFile:
        if self.model_type == ModelType.SIMPLE:
            return PPMSimpleInstructionsFile(self.instance._alphabet_levels, self.instance._order_bound, self.instance._input_sequence, results_file_path,
                                             self.instance._shortest_deterministic, self.instance._exclusion, self.instance._update_exclusion, self.instance._escape_method)
        elif self.model_type == ModelType.DECAY:
            return PPMDecayInstructionsFile(self.instance._alphabet_levels, self.instance._order_bound, self.instance._input_sequence, self.instance._input_time_sequence, results_file_path,
                                             self.instance._buffer_weight, self.instance._buffer_length_time, self.instance._buffer_length_items, self.instance._only_learn_from_buffer, self.instance._only_predict_from_buffer,
                                             self.instance._stm_weight, self.instance._stm_duration,
                                             self.instance._ltm_weight, self.instance._ltm_half_life, self.instance._ltm_asymptote,
                                             self.instance._noise, self.instance._seed)

    def run(self, instructions_file_path = ppmdecay_default_instructions_file_path(), results_file_path = ppmdecay_default_results_file_path()) -> ResultsMetaFile:
        instructions_file = self.to_instructions_file(results_file_path)
        instructions_file.write_instructions_file(instructions_file_path)
        model_output = invoke_model(instructions_file_path)

        results_meta_file = parse_results_meta_file(results_file_path)
        return results_meta_file
