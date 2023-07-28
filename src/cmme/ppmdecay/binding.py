from __future__ import annotations

import os
from abc import ABC
from pathlib import Path

import pandas as pd

from cmme.config import Config
from cmme.lib.instructions_file import InstructionsFile
from cmme.lib.results_file import ResultsFile
from cmme.ppmdecay.base import ModelType
from cmme.ppmdecay.util import list_to_str, str_to_list

# R_HOME specifies the R instance to use by rpy2.
# Needs to happen before any imports from rpy2
os.environ["R_HOME"] = str(Config().r_home())
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

PPM_RUN_FILEPATH = (Path(
    __file__).parent.parent.parent.parent.absolute() / "./res/wrappers/ppm-decay/ppmdecay_intermediate_script.R").resolve()


def invoke_model(instructions_file_path: Path) -> str:
    """

    :param instructions_file_path:
    :return: R console output
    """
    with open(PPM_RUN_FILEPATH) as f:
        r_file_contents = f.read()
    package = SignatureTranslatedAnonymousPackage(r_file_contents, "ppm-python-bridge")

    results_file_path = package.ppmdecay_intermediate_script(str(instructions_file_path))[0]

    return results_file_path


class PPMInstructionsFile(InstructionsFile, ABC):
    @staticmethod
    def _generate_instructions_file_path() -> str:
        pass

    def __init__(self, model_type: ModelType, alphabet_levels, order_bound, input_sequence, results_file_path: str):
        super().__init__()
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
            "results_file_path": [self._results_file_path]
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


class PPMSimpleInstructionsFile(PPMInstructionsFile):
    @classmethod
    def _save(cls, instructions_file: PPMSimpleInstructionsFile, file_path):
        data = {
            "model_type": [instructions_file._model_type.value],
            "alphabet_levels": [list_to_str(instructions_file._alphabet_levels)],
            "order_bound": [instructions_file._order_bound],
            "input_sequence": [instructions_file._input_sequence],
            "results_file_path": [instructions_file._results_file_path]
        }

        data.update({
            "shortest_deterministic": [instructions_file._shortest_deterministic],
            "exclusion": [instructions_file._exclusion],
            "update_exclusion": [instructions_file._update_exclusion],
            "escape": [instructions_file._escape_method.value]
        })

        df = pd.DataFrame.from_dict(data)
        df.to_feather(file_path)

    @classmethod
    def save(cls, instructions_file: PPMSimpleInstructionsFile,
             file_path: str = PPMInstructionsFile._generate_instructions_file_path()):
        cls._save(instructions_file, file_path)
    @staticmethod
    def load(file_path) -> PPMSimpleInstructionsFile:
        df = pd.read_feather(file_path)

        alphabet_levels = df["alphabet_levels"]
        order_bound = df["order_bound"]
        input_sequence = df["input_sequence"]
        results_file_path = df["results_file_path"]
        shortest_deterministic = df["shortest_deterministic"]
        exclusion = df["exclusion"]
        update_exclusion = df["update_exclusion"]
        escape_method = df["escape_method"]

        obj = PPMSimpleInstructionsFile(alphabet_levels, order_bound, input_sequence, results_file_path,
                                        shortest_deterministic, exclusion, update_exclusion, escape_method)

        return obj

    def __init__(self, alphabet_levels, order_bound, input_sequence, results_file_path,
                 shortest_deterministic, exclusion, update_exclusion, escape_method):
        super().__init__(ModelType.SIMPLE, alphabet_levels, order_bound, input_sequence, results_file_path)

        self._shortest_deterministic = shortest_deterministic
        self._exclusion = exclusion
        self._update_exclusion = update_exclusion
        self._escape_method = escape_method


class PPMDecayInstructionsFile(PPMInstructionsFile):
    @classmethod
    def _save(cls, instructions_file: PPMDecayInstructionsFile, file_path):
        data = {
            "model_type": [instructions_file._model_type.value],
            "alphabet_levels": [list_to_str(instructions_file._alphabet_levels)],
            "order_bound": [instructions_file._order_bound],
            "input_sequence": [instructions_file._input_sequence],
            "results_file_path": [instructions_file._results_file_path]
        }

        data.update({
            "input_time_sequence": [instructions_file._input_time_sequence],
            "buffer_weight": [instructions_file._buffer_weight],
            "buffer_length_time": [instructions_file._buffer_length_time],
            "buffer_length_items": [instructions_file._buffer_length_items],
            "stm_weight": [instructions_file._stm_weight],
            "stm_duration": [instructions_file._stm_duration],
            "only_learn_from_buffer": [instructions_file._only_learn_from_buffer],
            "only_predict_from_buffer": [instructions_file._only_predict_from_buffer],
            "ltm_weight": [instructions_file._ltm_weight],
            "ltm_half_life": [instructions_file._ltm_half_life],
            "ltm_asymptote": [instructions_file._ltm_asymptote],
            "noise": [instructions_file._noise],
            "seed": [instructions_file._seed]
        })

        df = pd.DataFrame.from_dict(data)
        df.to_feather(file_path)

    @classmethod
    def save(cls, instructions_file: PPMDecayInstructionsFile,
             file_path: str = PPMInstructionsFile._generate_instructions_file_path()):
        cls._save(instructions_file, file_path)

    @staticmethod
    def load(file_path) -> InstructionsFile:
        df = pd.read_feather(file_path)

        alphabet_levels = df["alphabet_levels"]
        order_bound = df["order_bound"]
        input_sequence = df["input_sequence"]
        input_time_sequence = df["input_time_sequence"]
        results_file_path = df["results_file_path"]
        buffer_weight = df["buffer_weight"]
        buffer_length_time = df["buffer_length_time"]
        buffer_length_items = df["buffer_length_items"]
        only_learn_from_buffer = df["only_learn_from_buffer"]
        only_predict_from_buffer = df["only_predict_from_buffer"]
        stm_weight = df["stm_weight"]
        stm_duration = df["stm_duration"]
        ltm_weight = df["ltm_weight"]
        ltm_half_life = df["ltm_half_life"]
        ltm_asymptote = df["ltm_asymptote"]
        noise = df["noise"]
        seed = df["seed"]

        obj = PPMDecayInstructionsFile(alphabet_levels, order_bound, input_sequence, input_time_sequence,
                                       results_file_path,
                                       buffer_weight, buffer_length_time, buffer_length_items, only_learn_from_buffer,
                                       only_predict_from_buffer,
                                       stm_weight, stm_duration, ltm_weight, ltm_half_life, ltm_asymptote,
                                       noise, seed)

        return obj

    def __init__(self, alphabet_levels, order_bound, input_sequence, input_time_sequence, results_file_path,
                 buffer_weight, buffer_length_time, buffer_length_items, only_learn_from_buffer,
                 only_predict_from_buffer,
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
        self.trials = list(set(df["trial_idx"].tolist()))

    def df_by_trial(self, trial):
        if trial not in self.trials:
            raise ValueError("trial {} does not exist!".format(trial))
        return self.df[self.df["trial_idx"] == trial]

    def df_of_last_trial(self):
        return self.df_by_trial(self.trials[-1])


class PPMSimpleResultsFileData(ResultsFileData):
    def __init__(self, results_file_data_path, df):
        super().__init__(results_file_data_path, df)


class PPMDecayResultsFileData(ResultsFileData):
    def __init__(self, results_file_data_path, df):
        super().__init__(results_file_data_path, df)


class PPMResultsMetaFile(ResultsFile):
    @staticmethod
    def _generate_results_file_path() -> str:
        return "results-file.feather"

    @staticmethod
    def _save(results_file: PPMResultsMetaFile, file_path: str):
        meta_df = pd.DataFrame.from_dict({
            "model_type": results_file._model_type,
            "alphabet_levels": results_file._alphabet_levels,
            "instructions_file_path": results_file._instructions_file_path,
            "results_file_data_path": results_file._results_file_data_path
        })
        data_df = results_file.results_file_data

        meta_df.to_feather(file_path)
        data_df.df.to_feather(file_path.replace(".feather", ".data.feather"))

    @staticmethod
    def load(file_path: str) -> PPMResultsMetaFile:
        df = pd.read_feather(file_path)

        model_type = ModelType(df["model_type"][0])
        alphabet_levels = str_to_list(df["alphabet_levels"][0])
        instructions_file_path = df["instructions_file_path"][0]
        results_file_data_path = df["results_file_data_path"][0]

        if not os.path.exists(results_file_data_path):
            results_file_data_filename = os.path.basename(results_file_data_path)
            original_results_file_data_path = results_file_data_path
            results_file_data_path = os.path.join(os.path.dirname(file_path), results_file_data_filename)
            if not os.path.exists(results_file_data_path):
                raise ValueError("Could not locate {}!".format(original_results_file_data_path))

        return PPMResultsMetaFile(file_path, model_type, alphabet_levels, instructions_file_path,
                                  results_file_data_path)

    def __init__(self, results_file_meta_path, model_type: ModelType, alphabet_levels, instructions_file_path,
                 results_file_data_path):
        super().__init__()
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
