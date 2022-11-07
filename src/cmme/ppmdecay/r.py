import csv
import os
from pathlib import Path

from config import Config
from .instructions_file import InstructionsFile, PPMSimpleInstructionsFile, PPMDecayInstructionsFile
from .model import PPMInstance, PPMSimpleInstance, PPMDecayInstance, ModelType
from .results_file import ResultsFile
from .util import list_to_str, str_to_list


def write_instructions_file(instructions_file: InstructionsFile, instructions_file_path = "instru.csv"):
    # TODO use generator for instructions_file_path
    data = {
        "model_type": instructions_file._model_type.value,
        "alphabet_levels": list_to_str(instructions_file._alphabet_levels),
        "order_bound": instructions_file._order_bound,
        "input_sequence": list_to_str(instructions_file._input_sequence),
        "input_time_sequence": list_to_str(instructions_file._input_time_sequence),
        "results_file_path": instructions_file._results_file_path
    }
    if isinstance(instructions_file, PPMSimpleInstructionsFile):
        data.update({
            "shortest_deterministic": instructions_file._shortest_deterministic,
            "exclusion": instructions_file._exclusion,
            "update_exclusion": instructions_file._update_exclusion,
            "escape": instructions_file._escape_method.value
        })
    elif isinstance(instructions_file, PPMDecayInstructionsFile):
        data.update({
            "input_time_sequence": list_to_str(instructions_file._input_time_sequence),
            "buffer_weight": instructions_file._buffer_weight,
            "buffer_length_time": instructions_file._buffer_length_time,
            "buffer_length_items": instructions_file._buffer_length_items,
            "stm_weight": instructions_file._stm_weight,
            "stm_duration": instructions_file._stm_duration,
            "only_learn_from_buffer": instructions_file._only_learn_from_buffer,
            "only_predict_from_buffer": instructions_file._only_predict_from_buffer,
            "ltm_weight": instructions_file._ltm_weight,
            "ltm_half_life": instructions_file._ltm_half_life,
            "ltm_asymptote": instructions_file._ltm_asymptote,
            "noise": instructions_file._noise,
            "seed": instructions_file._seed
        })

    with open(instructions_file_path, "w") as f:
        csvwriter = csv.DictWriter(f, fieldnames=data.keys())
        csvwriter.writeheader()
        csvwriter.writerow(data)

    return instructions_file_path

# R_HOME specifies the R instance to use by rpy2.
# Needs to happen before any imports from rpy2
os.environ["R_HOME"] = str(Config().r_home())
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
PPM_RUN_FILEPATH = (Path(__file__).parent.absolute() / "../res/wrappers/ppm-decay/ppmdecay_intermediate_script.R").resolve()


def invoke_model(ppm_instance: PPMInstance, results_file_path = "resu.csv"):
    with open(PPM_RUN_FILEPATH) as f:
        r_file_contents = f.read()
    package = SignatureTranslatedAnonymousPackage(r_file_contents, "ppm-python-bridge")

    instructions_file = None
    if isinstance(ppm_instance, PPMSimpleInstance):
        instructions_file = PPMSimpleInstructionsFile(ppm_instance._alphabet_levels, ppm_instance._order_bound, ppm_instance._input_sequence, ppm_instance._input_time_sequence, results_file_path,
                 ppm_instance._shortest_deterministic, ppm_instance._exclusion, ppm_instance._update_exclusion, ppm_instance._escape_method)
    elif isinstance(ppm_instance, PPMDecayInstance):
        instructions_file = PPMDecayInstructionsFile(ppm_instance._alphabet_levels, ppm_instance._order_bound, ppm_instance._input_sequence, ppm_instance._input_time_sequence, results_file_path,
                 ppm_instance._buffer_weight, ppm_instance._buffer_length_time, ppm_instance._buffer_length_items, ppm_instance._only_learn_from_buffer, ppm_instance._only_predict_from_buffer,
                 ppm_instance._stm_weight, ppm_instance._stm_duration, ppm_instance._ltm_weight, ppm_instance._ltm_half_life, ppm_instance._ltm_asymptote, ppm_instance._noise,
                 ppm_instance._seed)
    instructions_file_path = write_instructions_file(instructions_file)

    results_file_path = str(package.ppmdecay_intermediate_script(instructions_file_path))

    results_file = parse_results_file(results_file_path)
    return results_file


def parse_results_file(results_file_path):
    with open(results_file_path, 'r') as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            model_type = ModelType(row["model_type"])
            alphabet_levels = str_to_list(row["alphabet_levels"])
            instructions_file_path = row["instructions_file_path"]
            results_file_data_path = row["results_file_data_path"]

    return ResultsFile(model_type, alphabet_levels, instructions_file_path, results_file_data_path)
