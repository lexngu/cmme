import csv
from abc import ABC
from pathlib import Path

from .model import ModelType
from .util.util import list_to_str


class InstructionsFile(ABC):
    def __init__(self, model_type: ModelType, alphabet_levels, order_bound, input_sequence, results_file_path):
        self._model_type = model_type
        self._alphabet_levels = alphabet_levels
        self._order_bound = order_bound

        self._input_sequence = input_sequence
        self._results_file_path = results_file_path

    def write_instructions_file(self, instructions_file_path: Path):
        # TODO use generator for instructions_file_path
        data = {
            "model_type": self._model_type.value,
            "alphabet_levels": list_to_str(self._alphabet_levels),
            "order_bound": self._order_bound,
            "input_sequence": list_to_str(self._input_sequence),
            "results_file_path": self._results_file_path
        }
        if isinstance(self, PPMSimpleInstructionsFile):
            data.update({
                "shortest_deterministic": self._shortest_deterministic,
                "exclusion": self._exclusion,
                "update_exclusion": self._update_exclusion,
                "escape": self._escape_method.value
            })
        elif isinstance(self, PPMDecayInstructionsFile):
            data.update({
                "input_time_sequence": list_to_str(self._input_time_sequence),
                "buffer_weight": self._buffer_weight,
                "buffer_length_time": self._buffer_length_time,
                "buffer_length_items": self._buffer_length_items,
                "stm_weight": self._stm_weight,
                "stm_duration": self._stm_duration,
                "only_learn_from_buffer": self._only_learn_from_buffer,
                "only_predict_from_buffer": self._only_predict_from_buffer,
                "ltm_weight": self._ltm_weight,
                "ltm_half_life": self._ltm_half_life,
                "ltm_asymptote": self._ltm_asymptote,
                "noise": self._noise,
                "seed": self._seed
            })

        with open(instructions_file_path, "w") as f:
            csvwriter = csv.DictWriter(f, fieldnames=data.keys())
            csvwriter.writeheader()
            csvwriter.writerow(data)

        return instructions_file_path


class PPMSimpleInstructionsFile(InstructionsFile):
    def __init__(self, alphabet_levels, order_bound, input_sequence, results_file_path,
                 shortest_deterministic, exclusion, update_exclusion, escape_method):
        super().__init__(ModelType.SIMPLE, alphabet_levels, order_bound, input_sequence, results_file_path)

        self._shortest_deterministic = shortest_deterministic
        self._exclusion = exclusion
        self._update_exclusion = update_exclusion
        self._escape_method = escape_method


class PPMDecayInstructionsFile(InstructionsFile):
    def __init__(self, alphabet_levels, order_bound, input_sequence, input_time_sequence, results_file_path,
                 buffer_weight, buffer_length_time, buffer_length_items, only_learn_from_buffer, only_predict_from_buffer,
                 stm_weight, stm_duration, ltm_weight, ltm_half_life, ltm_asymptote,
                 noise, seed):
        super().__init__(ModelType.DECAY, alphabet_levels, order_bound, input_sequence, results_file_path)

        self._input_time_sequence = input_time_sequence
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