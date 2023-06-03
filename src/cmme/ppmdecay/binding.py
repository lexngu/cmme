import os
from abc import ABC
from pathlib import Path

import pandas as pd

from cmme.config import Config
from cmme.ppmdecay.base import ModelType
from cmme.ppmdecay.util import list_to_str, str_to_list
# R_HOME specifies the R instance to use by rpy2.
# Needs to happen before any imports from rpy2
os.environ["R_HOME"] = str(Config().r_home())
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
PPM_RUN_FILEPATH = (Path(__file__).parent.parent.parent.parent.absolute() / "./res/wrappers/ppm-decay/ppmdecay_intermediate_script.R").resolve()


def invoke_model(instructions_file_path : Path):
    """

    :param instructions_file_path:
    :return: R console output
    """
    with open(PPM_RUN_FILEPATH) as f:
        r_file_contents = f.read()
    package = SignatureTranslatedAnonymousPackage(r_file_contents, "ppm-python-bridge")

    results_file_path = str(package.ppmdecay_intermediate_script(str(instructions_file_path)))

    return results_file_path


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
            "model_type": [self._model_type.value],
            "alphabet_levels": [list_to_str(self._alphabet_levels)],
            "order_bound": [self._order_bound],
            "input_sequence": [self._input_sequence],
            "results_file_path": [str(self._results_file_path)]
        }
        if isinstance(self, PPMSimpleInstructionsFile):
            data.update({
                "shortest_deterministic": [self._shortest_deterministic],
                "exclusion": [self._exclusion],
                "update_exclusion": [self._update_exclusion],
                "escape": [self._escape_method.value]
            })
        elif isinstance(self, PPMDecayInstructionsFile):
            data.update({
                "input_time_sequence": [self._input_time_sequence],
                "buffer_weight": [self._buffer_weight],
                "buffer_length_time": [self._buffer_length_time],
                "buffer_length_items": [self._buffer_length_items],
                "stm_weight": [self._stm_weight],
                "stm_duration": [self._stm_duration],
                "only_learn_from_buffer": [self._only_learn_from_buffer],
                "only_predict_from_buffer": [self._only_predict_from_buffer],
                "ltm_weight": [self._ltm_weight],
                "ltm_half_life": [self._ltm_half_life],
                "ltm_asymptote": [self._ltm_asymptote],
                "noise": [self._noise],
                "seed": [self._seed]
            })

        df = pd.DataFrame.from_dict(data)
        df.to_feather(instructions_file_path)

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


class ResultsFileData(ABC):
    def __init__(self, results_file_data_path, df):
        self.results_file_data_path = results_file_data_path
        self.df = df


class PPMSimpleResultsFileData(ResultsFileData):
    def __init__(self, results_file_data_path, df):
        super().__init__(results_file_data_path, df)


class PPMDecayResultsFileData(ResultsFileData):
    def __init__(self, results_file_data_path, df):
        super().__init__(results_file_data_path, df)


class ResultsMetaFile:
    def __init__(self, results_file_meta_path, model_type: ModelType, alphabet_levels, instructions_file_path, results_file_data_path):
        self.results_file_meta_path = results_file_meta_path
        self._model_type = model_type
        self._alphabet_levels = alphabet_levels
        self._instructions_file_path = instructions_file_path
        self._results_file_data_path = results_file_data_path
        if model_type == ModelType.SIMPLE:
            self.results_file_data = self._parse_ppm_simple_results_file_data()
        else:
            self.results_file_data = self._parse_ppm_decay_results_file_data()

    def _parse_ppm_simple_results_file_data(self):
        df = pd.read_feather(self._results_file_data_path)
        return PPMSimpleResultsFileData(self._results_file_data_path, df)

    def _parse_ppm_decay_results_file_data(self):
        df = pd.read_feather(self._results_file_data_path)
        return PPMDecayResultsFileData(self._results_file_data_path, df)

import os

def parse_results_meta_file(results_file_meta_path : Path):
    df = pd.read_feather(results_file_meta_path)

    model_type = ModelType(df["model_type"][0])
    alphabet_levels = str_to_list(df["alphabet_levels"][0])
    instructions_file_path = df["instructions_file_path"][0]
    results_file_data_path = df["results_file_data_path"][0]

    if not os.path.exists(results_file_data_path):
        results_file_data_filename = os.path.basename(results_file_data_path)
        original_results_file_data_path = results_file_data_path
        results_file_data_path = os.path.join(os.path.dirname(str(results_file_meta_path)), results_file_data_filename)
        if not os.path.exists(results_file_data_path):
            raise ValueError("Could not locate {}!".format(original_results_file_data_path))

    return ResultsMetaFile(results_file_meta_path, model_type, alphabet_levels, instructions_file_path, results_file_data_path)
