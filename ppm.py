import os
from .config import Config
import csv
import pandas as pd
from pathlib import Path
from enum import Enum

# R_HOME specifies the R instance to use by rpy2.
# Needs to happen before any imports from rpy2
os.environ["R_HOME"] = str(Config().r_home())
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


class PPMEscapeMethod(Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    AX = "ax"

class PPMSimpleInputParameters:
    """This class represents all parameters of any PPM-Simple instance."""

    def __init__(self, order_bound, shortest_deterministic, exclusion, update_exclusion, escape: PPMEscapeMethod, alphabet_levels):
        self._order_bound = order_bound
        self._shortest_deterministic = shortest_deterministic
        self._exclusion = exclusion
        self._update_exclusion = update_exclusion
        self._escape = escape
        self._alphabet_levels = alphabet_levels

        self._results_file_path = None

        self._sequence = None

    def with_sequence(self, sequence):
        self._sequence = sequence
        return self

    def with_results_file_path(self, results_file_path: Path):
        self._results_file_path = results_file_path
        return self

    def write_csv(self, file_path: Path):
        params = {
            "model_as_string": str(self),
            "model": "SIMPLE",
            "order_bound": self._order_bound,
            "alphabet_levels": ", ".join(map(str, self._alphabet_levels)),
            "shortest_deterministic": self._shortest_deterministic,
            "exclusion": self._exclusion,
            "update_exclusion": self._update_exclusion,
            "escape": self._escape.value,

            "input_sequence": ", ".join(map(str, self._sequence)),
            "results_file_path": str(self._results_file_path)
        }

        with open(file_path, 'w') as f:
            csvwriter = csv.DictWriter(f, fieldnames=params.keys())
            csvwriter.writeheader()
            csvwriter.writerow(params)

        return file_path

    def read_csv(filename): # TODO implement import
        return PPMSimpleInputParameters()

    def __str__(self):
        return "PPM-Decay(model=SIMPLE, order_bound={}, shortest_deterministic={}, exclusion={}, update_exclusion={}, escape={}, alphabet_levels={})".format(
            self._order_bound, self._shortest_deterministic, self._exclusion, self._update_exclusion, self._escape, self._alphabet_levels
        )

class PPMDecayInputParameters:
    """This class represents all parameters of any PPM-Decay instance."""

    def __init__(self, order_bound, ltm_weight, ltm_half_life, ltm_asymptote, noise, stm_weight,
                 stm_duration, buffer_weight, buffer_length_time, buffer_length_items, only_learn_from_buffer,
                 only_predict_from_buffer, seed, alphabet_levels):
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
        self._alphabet_levels = alphabet_levels

        self._results_file_path = None

        self._sequence = None
        self._time_sequence = None

    def with_sequence(self, sequence):
        self._sequence = sequence
        return self

    def with_time_sequence(self, time_sequence: list):
        self._time_sequence = time_sequence
        return self

    def with_results_file_path(self, results_file_path: Path):
        self._results_file_path = results_file_path
        return self

    def write_csv(self, file_path: Path):
        params = {
            "model_as_string": str(self),
            "model": "DECAY",
            "alphabet_levels": ", ".join(map(str, self._alphabet_levels)),
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

            "input_sequence": ", ".join(map(str, self._sequence)),
            "input_time_sequence": ", ".join(map(str, self._time_sequence)),
            "results_file_path": str(self._results_file_path)
        }

        with open(file_path, 'w') as f:
            csvwriter = csv.DictWriter(f, fieldnames=params.keys())
            csvwriter.writeheader()
            csvwriter.writerow(params)

        return file_path

    def read_csv(filename): # TODO implement import
        return PPMDecayInputParameters()

    def __str__(self):
        return "PPM-Decay(model=DECAY, order_bound={}, buffer_weight={}, buffer_length_time={}, buffer_length_items={}, only_learn_from_buffer={}, only_predict_from_buffer={}, stm_weight={}, stm_duration={}, ltm_weight={}, ltm_half_life={}, ltm_asymptote={}, noise={}, seed={}, alphabet_levels={})".format(
            self._order_bound, self._buffer_weight, self._buffer_length_time, self._buffer_length_items, self._only_learn_from_buffer, self._only_predict_from_buffer, self._stm_weight, self._stm_duration, self._ltm_weight, self._ltm_half_life, self._ltm_asymptote, self._noise, self._seed, self._alphabet_levels
        )

class PPMOutputParameters:
    """This class represents an PPM output."""

    def __init__(self, source_file_path: Path, data_frame: pd.DataFrame):
        self.source_file_path = source_file_path
        self.data_frame = pd.DataFrame.copy(data_frame)

        self.input_sequence = self.data_frame.drop_duplicates(subset=['pos'])['symbol'].tolist()

    def from_csv(file_path):
        """Imports PPM output (.csv file)"""

        csv_data = pd.read_csv(file_path)
        return PPMOutputParameters(file_path, csv_data)


class PPMInstance:
    """This class represents a PPM instance."""

    PPM_RUN_FILEPATH = (Path(__file__).parent.absolute() / "./res/wrappers/ppm-decay/ppm_run.R").resolve()

    def __init__(self, ppmInputParameters, instructions_file_path: Path, results_file_path: Path):
        self._ppmInputParameters = ppmInputParameters
        self.instructions_file_path = instructions_file_path
        self.results_file_path = results_file_path

        self._r_script = PPMInstance._init_r_script()

    def _init_r_script():
        with open(PPMInstance.PPM_RUN_FILEPATH) as f:
            r_file_contents = f.read()
        package = SignatureTranslatedAnonymousPackage(r_file_contents, "ppm-python-bridge")
        return package

    def _generate_time_sequence(sequence):
        return list(range(1, len(sequence) + 1))

    def observe(self, sequence, time_sequence = None): # TODO In the original code, there are additional parameters for model_seq(); implement?
        """ If [], and sequence is a list, a list with increasing numbers is generated, i.e. [1, 2, ..., len(sequence)]"""

        if isinstance(self._ppmInputParameters, PPMDecayInputParameters):
            if time_sequence is None or not isinstance(time_sequence, list):
                raise ValueError("time_sequence invalid! Provide [], or any other list.")
            if len(time_sequence) == 0:
                time_sequence = PPMInstance._generate_time_sequence(sequence)
            if len(time_sequence) != len(sequence):
                raise ValueError("Length of sequence and time_sequence must be equal.")

        self._ppmInputParameters.with_sequence(sequence).with_results_file_path(self.results_file_path)
        if isinstance(self._ppmInputParameters, PPMDecayInputParameters):
            self._ppmInputParameters.with_time_sequence(time_sequence)
        self._ppmInputParameters.write_csv(self.instructions_file_path)

        result = self._r_script.run_ppm(str(self.instructions_file_path))
        print(result)
        return self.results_file_path  # ToDo: return R output, which should contain the results_file_path.

class PPMSimpleInstanceBuilder:
    def __init__(self):
        self._order_bound = 10
        self._shortest_deterministic = True
        self._exclusion = True
        self._update_exclusion = True
        self._escape = PPMEscapeMethod.C
        self._alphabet_levels = {}

    def alphabet_levels(self, alphabet_levels: list):
        if not len(alphabet_levels) == len(set(alphabet_levels)):
            raise ValueError("Invalid alphabet_levels! Value must contain unique elements!")

        self._alphabet_levels = alphabet_levels

        return self

    def order_bound(self, order_bound):
        self._order_bound = order_bound
        return self

    def shortest_deterministic(self, shortest_deterministic):
        self._shortest_deterministic = shortest_deterministic
        return self

    def exclusion(self, exclusion):
        self._exclusion = exclusion
        return self

    def update_exclusion(self, update_exclusion):
        self._update_exclusion = update_exclusion
        return self

    def escape(self, escape: PPMEscapeMethod):
        if not isinstance(escape, PPMEscapeMethod):
            raise ValueError("escape invalid! It must be member of PPMEscapeMethod enum.")
        self._escape = escape
        return self

    def with_instructions_file_path(self, instructions_file_path: Path):
        self.instructions_file_path = instructions_file_path

        return self

    def with_results_file_path(self, results_file_path: Path):
        self.results_file_path = results_file_path

        return self

    def build(self):
        if self.instructions_file_path is None:
            raise Exception("instructions_file_path required!")
        if self.results_file_path is None:
            raise Exception("results_file_path required!")

        ppmInputParameters = PPMSimpleInputParameters(self._order_bound, self._shortest_deterministic, self._exclusion, self._update_exclusion, self._escape, self._alphabet_levels)
        return PPMInstance(ppmInputParameters, self.instructions_file_path, self.results_file_path)


class PPMDecayInstanceBuilder:
    def __init__(self):
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
        self._seed = 1 # TODO make random as in original source?
        self._alphabet_levels = {}

        self.instructions_file_path = None
        self.results_file_path = None

    def alphabet_levels(self, alphabet_levels: list):
        if not len(alphabet_levels) == len(set(alphabet_levels)):
            raise ValueError("Invalid alphabet_levels! Value must contain unique elements!")

        self._alphabet_levels = alphabet_levels

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

    def with_instructions_file_path(self, instructions_file_path: Path):
        self.instructions_file_path = instructions_file_path

        return self

    def with_results_file_path(self, results_file_path: Path):
        self.results_file_path = results_file_path

        return self

    def build(self):
        if self.instructions_file_path is None:
            raise Exception("instructions_file_path required!")
        if self.results_file_path is None:
            raise Exception("results_file_path required!")

        ppmInputParameters = PPMDecayInputParameters(self._order_bound, self._ltm_weight,
                                                     self._ltm_half_life, self._ltm_asymptote, self._noise,
                                                     self._stm_weight, self._stm_duration, self._buffer_weight,
                                                     self._buffer_length_time, self._buffer_length_items,
                                                     self._only_learn_from_buffer, self._only_predict_from_buffer,
                                                     self._seed,
                                                     self._alphabet_levels)
        return PPMInstance(ppmInputParameters, self.instructions_file_path, self.results_file_path)