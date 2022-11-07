from abc import ABC

from .model import ModelType


class InstructionsFile(ABC):
    def __init__(self, model_type: ModelType, alphabet_levels, order_bound, input_sequence, input_time_sequence, results_file_path):
        self._model_type = model_type
        self._alphabet_levels = alphabet_levels
        self._order_bound = order_bound

        self._input_sequence = input_sequence
        self._input_time_sequence = input_time_sequence
        self._results_file_path = results_file_path


class PPMSimpleInstructionsFile(InstructionsFile):
    def __init__(self, alphabet_levels, order_bound, input_sequence, input_time_sequence, results_file_path,
                 shortest_deterministic, exclusion, update_exclusion, escape_method):
        super().__init__(ModelType.SIMPLE, alphabet_levels, order_bound, input_sequence, input_time_sequence, results_file_path)
        self._shortest_deterministic = shortest_deterministic
        self._exclusion = exclusion
        self._update_exclusion = update_exclusion
        self._escape_method = escape_method


class PPMDecayInstructionsFile(InstructionsFile):
    def __init__(self, alphabet_levels, order_bound, input_sequence, input_time_sequence, results_file_path,
                 buffer_weight, buffer_length_time, buffer_length_items, only_learn_from_buffer, only_predict_from_buffer,
                 stm_weight, stm_duration, ltm_weight, ltm_half_life, ltm_asymptote, noise,
                 seed):
        super().__init__(ModelType.DECAY, alphabet_levels, order_bound, input_sequence, input_time_sequence, results_file_path)
        self._buffer_weight = buffer_weight
        self._buffer_length_time = buffer_length_time
        self._buffer_length_items = buffer_length_items
        self._only_learn_from_buffer = only_learn_from_buffer
        self._only_predict_from_buffer = only_predict_from_buffer
        self._stm_weight = stm_weight
        self._stm_duration = stm_duration
        self._ltm_weight = ltm_weight
        self._ltm_half_life = ltm_half_life
        self._ltm_asymptote = ltm_asymptote
        self._noise = noise

        self._seed = seed