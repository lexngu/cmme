from __future__ import annotations

import os
from abc import ABC
from pathlib import Path
from typing import Union

import pandas as pd

from cmme.config import Config
from cmme.lib.instructions_file import InstructionsFile
from cmme.lib.results_file import ResultsFile
from cmme.lib.util import nparray_to_list
from cmme.ppmdecay.base import ModelType, EscapeMethod
from cmme.ppmdecay.util import list_to_str, str_to_list

# R_HOME specifies the R instance to use by rpy2.
# Needs to happen before any imports from rpy2
os.environ["R_HOME"] = str(Config().r_home())
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

PPM_RUN_FILEPATH = (Path(
    __file__).parent.parent.parent.parent.absolute() / "./res/wrappers/ppm-decay/ppmdecay_intermediate_script.R").resolve()


def invoke_model(instructions_file_path: Union[str, Path]) -> str:
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
    def __init__(self, model_type: ModelType, alphabet_levels, order_bound, input_sequence):
        super().__init__()
        self.model_type = model_type
        self.alphabet_levels = alphabet_levels
        self.order_bound = order_bound

        self.input_sequence = input_sequence


class PPMSimpleInstructionsFile(PPMInstructionsFile):
    @classmethod
    def save(cls, instructions_file: PPMSimpleInstructionsFile, instructions_file_path: Union[str, Path],
             results_file_path: Union[str, Path] = None):
        data = {
            "model_type": [instructions_file.model_type.value],
            "alphabet_levels": [list_to_str(instructions_file.alphabet_levels)],
            "order_bound": [instructions_file.order_bound],
            "input_sequence": [instructions_file.input_sequence],
            "results_file_path": [str(results_file_path)] if results_file_path is not None else [""]
        }

        data.update({
            "shortest_deterministic": [instructions_file.shortest_deterministic],
            "exclusion": [instructions_file.exclusion],
            "update_exclusion": [instructions_file.update_exclusion],
            "escape": [instructions_file.escape_method.value]
        })

        df = pd.DataFrame.from_dict(data)
        df.to_feather(instructions_file_path)

    @staticmethod
    def load(file_path: Union[str, Path]) -> PPMSimpleInstructionsFile:
        df = pd.read_feather(file_path)

        alphabet_levels = str_to_list(df["alphabet_levels"][0])
        order_bound = int(df["order_bound"][0])
        input_sequence = nparray_to_list(df["input_sequence"][0])
        shortest_deterministic = df["shortest_deterministic"][0]
        exclusion = df["exclusion"][0]
        update_exclusion = df["update_exclusion"][0]
        escape_method = EscapeMethod(df["escape"][0])

        obj = PPMSimpleInstructionsFile(alphabet_levels, order_bound, input_sequence,
                                        shortest_deterministic, exclusion, update_exclusion, escape_method)

        return obj

    def __init__(self, alphabet_levels, order_bound, input_sequence,
                 shortest_deterministic, exclusion, update_exclusion, escape_method):
        super().__init__(ModelType.SIMPLE, alphabet_levels, order_bound, input_sequence)

        self.shortest_deterministic = shortest_deterministic
        self.exclusion = exclusion
        self.update_exclusion = update_exclusion
        self.escape_method = escape_method


class PPMDecayInstructionsFile(PPMInstructionsFile):
    @classmethod
    def save(cls, instructions_file: PPMDecayInstructionsFile, instructions_file_path: Union[str, Path],
             results_file_path: Union[str, Path] = None):
        data = {
            "model_type": [instructions_file.model_type.value],
            "alphabet_levels": [list_to_str(instructions_file.alphabet_levels)],
            "order_bound": [instructions_file.order_bound],
            "input_sequence": [instructions_file.input_sequence],
            "results_file_path": [str(results_file_path)] if results_file_path is not None else [""]
        }

        data.update({
            "input_time_sequence": [instructions_file.input_time_sequence],
            "buffer_weight": [instructions_file.buffer_weight],
            "buffer_length_time": [instructions_file.buffer_length_time],
            "buffer_length_items": [instructions_file.buffer_length_items],
            "stm_weight": [instructions_file.stm_weight],
            "stm_duration": [instructions_file.stm_duration],
            "only_learn_from_buffer": [instructions_file.only_learn_from_buffer],
            "only_predict_from_buffer": [instructions_file.only_predict_from_buffer],
            "ltm_weight": [instructions_file.ltm_weight],
            "ltm_half_life": [instructions_file.ltm_half_life],
            "ltm_asymptote": [instructions_file.ltm_asymptote],
            "noise": [instructions_file.noise],
            "seed": [instructions_file.seed]
        })

        df = pd.DataFrame.from_dict(data)
        df.to_feather(instructions_file_path)

    @staticmethod
    def load(file_path: Union[str, Path]) -> InstructionsFile:
        df = pd.read_feather(file_path)

        alphabet_levels = str_to_list(df["alphabet_levels"][0])
        order_bound = df["order_bound"][0]
        input_sequence = nparray_to_list(df["input_sequence"][0])
        input_time_sequence = nparray_to_list(df["input_time_sequence"][0])
        buffer_weight = df["buffer_weight"][0]
        buffer_length_time = df["buffer_length_time"][0]
        buffer_length_items = df["buffer_length_items"][0]
        only_learn_from_buffer = df["only_learn_from_buffer"][0]
        only_predict_from_buffer = df["only_predict_from_buffer"][0]
        stm_weight = df["stm_weight"][0]
        stm_duration = df["stm_duration"][0]
        ltm_weight = df["ltm_weight"][0]
        ltm_half_life = df["ltm_half_life"][0]
        ltm_asymptote = df["ltm_asymptote"][0]
        noise = df["noise"][0]
        seed = df["seed"][0]

        obj = PPMDecayInstructionsFile(alphabet_levels, order_bound, input_sequence, input_time_sequence,
                                       buffer_weight, buffer_length_time, buffer_length_items, only_learn_from_buffer,
                                       only_predict_from_buffer,
                                       stm_weight, stm_duration, ltm_weight, ltm_half_life, ltm_asymptote,
                                       noise, seed)

        return obj

    def __init__(self, alphabet_levels, order_bound, input_sequence, input_time_sequence,
                 buffer_weight, buffer_length_time, buffer_length_items, only_learn_from_buffer,
                 only_predict_from_buffer,
                 stm_weight, stm_duration, ltm_weight, ltm_half_life, ltm_asymptote,
                 noise, seed):
        super().__init__(ModelType.DECAY, alphabet_levels, order_bound, input_sequence)

        self.input_time_sequence = input_time_sequence
        self.buffer_weight = buffer_weight
        self.buffer_length_time = buffer_length_time
        self.buffer_length_items = buffer_length_items
        self.only_learn_from_buffer = only_learn_from_buffer
        self.only_predict_from_buffer = only_predict_from_buffer
        self.stm_weight = stm_weight
        self.stm_duration = stm_duration
        self.ltm_weight = ltm_weight
        self.ltm_half_life = ltm_half_life
        self.ltm_asymptote = ltm_asymptote
        self.noise = noise

        self.seed = seed


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
    def save(results_file: PPMResultsMetaFile, file_path: Union[str, Path]):
        meta_df = pd.DataFrame.from_dict({
            "model_type": results_file.model_type,
            "alphabet_levels": results_file.alphabet_levels,
            "instructions_file_path": str(results_file.instructions_file_path),
            "results_file_data_path": str(results_file.results_file_data_path)
        })
        data_df = results_file.results_file_data

        meta_df.to_feather(file_path)
        data_df.df.to_feather(str(file_path).replace(".feather", ".data.feather"))

    @staticmethod
    def load(file_path: Union[str, Path]) -> PPMResultsMetaFile:
        df = pd.read_feather(file_path)

        model_type = ModelType(df["model_type"][0])
        alphabet_levels = str_to_list(df["alphabet_levels"][0])
        instructions_file_path = str(df["instructions_file_path"][0])
        results_file_data_path = str(df["results_file_data_path"][0])

        if not os.path.exists(results_file_data_path):
            results_file_data_filename = os.path.basename(results_file_data_path)
            original_results_file_data_path = results_file_data_path
            results_file_data_path = os.path.join(os.path.dirname(file_path), results_file_data_filename)
            if not os.path.exists(results_file_data_path):
                raise ValueError("Could not locate {}!".format(original_results_file_data_path))

        return PPMResultsMetaFile(file_path, model_type, alphabet_levels, instructions_file_path,
                                  results_file_data_path)

    def __init__(self, results_file_meta_path: Path, model_type: ModelType, alphabet_levels, instructions_file_path,
                 results_file_data_path):
        super().__init__()
        self.results_file_meta_path = results_file_meta_path
        self.model_type = model_type
        self.alphabet_levels = alphabet_levels
        self.instructions_file_path = instructions_file_path
        self.results_file_data_path = results_file_data_path
        if model_type == ModelType.SIMPLE:
            self.results_file_data = self._parse_ppm_simple_results_file_data()
        else:
            self.results_file_data = self._parse_ppm_decay_results_file_data()

    def _parse_ppm_simple_results_file_data(self):
        df = pd.read_feather(self.results_file_data_path)
        return PPMSimpleResultsFileData(self.results_file_data_path, df)

    def _parse_ppm_decay_results_file_data(self):
        df = pd.read_feather(self.results_file_data_path)
        return PPMDecayResultsFileData(self.results_file_data_path, df)
