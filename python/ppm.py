import csv
import pandas as pd
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from pathlib import Path

class PPMInputParameters:
    def __init__(self, alphabet_size, order_bound, ltm_weight, ltm_half_life, ltm_asymptote, noise, stm_weight,
                 stm_duration, buffer_weight, buffer_length_time, buffer_length_items, only_learn_from_buffer,
                 only_predict_from_buffer, seed, debug_smooth, debug_decay, alphabet_levels):
        self._alphabet_size = alphabet_size
        self._order_bound = order_bound
        self._ltm_weight = ltm_weight
        self._ltm_half_life = ltm_half_life
        self._ltm_asymptote = ltm_asymptote
        self._noise = noise
        self._stm_weight = stm_weight
        self._stm_duration = stm_duration
        self._buffer_weight = buffer_weight
        self._buffer_length_time = buffer_length_time
        self._buffer_length_items = buffer_length_items
        self._only_learn_from_buffer = only_learn_from_buffer
        self._only_predict_from_buffer = only_predict_from_buffer
        self._seed = seed
        self._debug_smooth = debug_smooth
        self._debug_decay = debug_decay
        self._alphabet_levels = alphabet_levels

        self._sequence = None

    def with_sequence(self, sequence):
        self._sequence = sequence
        return self

    def with_output_parameters_file_path(self, output_parameters_file_path):
        self._output_parameters_file_path = output_parameters_file_path
        return self

    def write_csv(self, filename):
        params = {
            "alphabet_size": self._alphabet_size,
            "alphabet_levels": self._alphabet_levels,
            "order_bound": self._order_bound,
            "ltm_weight": self._ltm_weight,
            "ltm_half_life": self._ltm_half_life,
            "ltm_asymptote": self._ltm_asymptote,
            "noise": self._noise,
            "stm_weight": self._stm_weight,
            "stm_duration": self._stm_duration,
            "buffer_weight": self._buffer_weight,
            "buffer_length_time": self._buffer_length_time,
            "buffer_length_items": self._buffer_length_items,
            "only_learn_from_buffer": self._only_learn_from_buffer,
            "only_predict_from_buffer": self._only_predict_from_buffer,
            "seed": self._seed,
            "debug_smooth": self._debug_smooth,
            "debug_decay": self._debug_decay,

            "input_sequence": self._sequence,
            "input_time_sequence": list(range(1, len(self._sequence) + 1)),  # ToDo: remove hard-code
            "output_parameters_file_path": self._output_parameters_file_path
        }

        with open(filename, 'w') as f:
            csvwriter = csv.DictWriter(f, fieldnames=params.keys())
            csvwriter.writeheader()
            csvwriter.writerow(params)

        return filename

    def read_csv(filename):
        return PPMInputParameters()


class PPMOutputParameters:
    def __init__(self, source_file_path, data_frame):
        self.source_file_path = source_file_path
        self.data_frame = pd.DataFrame.copy(data_frame)

        self.input_sequence = self.data_frame.drop_duplicates(subset=['symbol_idx'])['observation'].tolist()
        self.alphabet_size = len(self.data_frame['distribution_idx'].unique())

    def from_csv(file_path):
        csv_data = pd.read_csv(file_path)
        return PPMOutputParameters(file_path, csv_data)


class PPMInstance:
    PPM_RUN_FILEPATH = (Path( __file__ ).parent.absolute() / "./res/r/ppm_run.R").resolve()

    def __init__(self, ppmInputParameters, model_io_paths):
        self._ppmInputParameters = ppmInputParameters
        self._model_io_paths = model_io_paths

        self._r_script = PPMInstance._init_r_script()

    def _init_r_script():
        with open(PPMInstance.PPM_RUN_FILEPATH) as f:
            r_file_contents = f.read()
        package = SignatureTranslatedAnonymousPackage(r_file_contents, "ppm-python-bridge")
        return package

    def observe(self, sequence):
        input_file_path = self._model_io_paths["input_file_path"] + ".csv"
        output_file_path = self._model_io_paths["output_file_path"] + ".csv"
        self._ppmInputParameters.with_sequence(sequence).with_output_parameters_file_path(output_file_path).write_csv(
            input_file_path)

        result = self._r_script.run_ppm(input_file_path)
        return output_file_path  # ToDo: return R output, which should contain the output_file_path.


class PPMInstanceBuilder:
    def __init__(self, model_io_paths):
        self._alphabet_size = None
        self._order_bound = 10
        self._ltm_weight = 1
        self._ltm_half_life = 10
        self._ltm_asymptote = 0
        self._noise = 0
        self._stm_weight = 1
        self._stm_duration = 0
        self._buffer_weight = 1
        self._buffer_length_time = 0
        self._buffer_length_items = 0
        self._only_learn_from_buffer = False
        self._only_predict_from_buffer = False
        self._seed = 1
        self._debug_smooth = False
        self._debug_decay = False
        self._alphabet_levels = []

        self._model_io_paths = model_io_paths

    def alphabet_size(self, alphabet_size):
        if alphabet_size > 0:
            self._alphabet_size = alphabet_size
        else:
            raise ValueError("Invalid alphabet_size! Value must be > 0.")
        return self

    def alphabet_levels(self, alphabet_levels):
        if isinstance(alphabet_levels, set):
            self._alphabet_levels = alphabet_levels
            self.alphabet_size(len(alphabet_levels))
        else:
            raise ValueError("Invalid alphabet_levels! Value must be a set.")
        return self

    def order_bound(self, order_bound):
        self._order_bound = order_bound
        return self

    def ltm_weight(self, ltm_weight):
        self._ltm_weight = ltm_weight
        return self

    def ltm_half_life(self, ltm_half_life):
        self._ltm_half_life = ltm_half_life
        return self

    def ltm_asymptote(self, ltm_asymptote):
        self._ltm_asymptote = ltm_asymptote
        return self

    def noise(self, noise):
        self._noise = noise
        return self

    def stm_weight(self, stm_weight):
        self._stm_weight = stm_weight
        return self

    def stm_duration(self, stm_duration):
        self._stm_duration = stm_duration
        return self

    def buffer_weight(self, buffer_weight):
        self._buffer_weight = buffer_weight
        return self

    def buffer_length_time(self, buffer_length_time):
        self._buffer_length_time = buffer_length_time
        return self

    def buffer_length_items(self, buffer_length_items):
        self._buffer_length_items = buffer_length_items
        return self

    def only_learn_from_buffer(self, only_learn_from_buffer):
        self._only_learn_from_buffer = only_learn_from_buffer
        return self

    def only_predict_from_buffer(self, only_predict_from_buffer):
        self._only_predict_from_buffer = only_predict_from_buffer
        return self

    def seed(self, seed):
        self._seed = seed
        return self

    def debug_smooth(self, debug_smooth):
        self._debug_smooth = debug_smooth
        return self

    def debug_decay(self, debug_decay):
        self._debug_decay = debug_decay
        return self

    def build(self):
        if self._alphabet_size != None and self._alphabet_size > 0:
            ppmInputParameters = PPMInputParameters(self._alphabet_size, self._order_bound, self._ltm_weight,
                                                    self._ltm_half_life, self._ltm_asymptote, self._noise,
                                                    self._stm_weight, self._stm_duration, self._buffer_weight,
                                                    self._buffer_length_time, self._buffer_length_items,
                                                    self._only_learn_from_buffer, self._only_predict_from_buffer,
                                                    self._seed, self._debug_smooth, self._debug_decay,
                                                    self._alphabet_levels)
            return PPMInstance(ppmInputParameters, self._model_io_paths)
        else:
            raise ValueError("Invalid alphabet_size! Value must be > 0.")
