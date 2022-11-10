from cmme.ppmdecay.instance import ModelType, PPMSimpleInstance, PPMDecayInstance
from cmme.ppmdecay.instructions_file import InstructionsFile, PPMSimpleInstructionsFile, PPMDecayInstructionsFile
from cmme.ppmdecay.util.r import invoke_model
from cmme.ppmdecay.results_file import ResultsFileData, ResultsMetaFile, parse_results_meta_file
from cmme.ppmdecay.util.util import ppmdecay_default_instructions_file_path, ppmdecay_default_results_file_path


class PPMModel:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type

        if model_type == ModelType.SIMPLE:
            self.instance = PPMSimpleInstance()
        elif model_type == ModelType.DECAY:
            self.instance = PPMDecayInstance()

    def to_instructions_file(self, instructions_file_path = ppmdecay_default_instructions_file_path(), results_file_path = ppmdecay_default_results_file_path()) -> InstructionsFile:
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
        instructions_file = self.to_instructions_file(instructions_file_path, results_file_path)
        instructions_file.write_instructions_file(instructions_file_path)
        model_output = invoke_model(instructions_file_path)

        results_meta_file = parse_results_meta_file(results_file_path)
        return results_meta_file
