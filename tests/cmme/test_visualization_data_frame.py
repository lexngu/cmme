import tempfile
from pathlib import Path

from cmme.drex.base import DistributionType, UnprocessedPrior
from cmme.drex.model import DREXInstructionBuilder
from cmme.lib.util import drex_default_instructions_file_path, drex_default_results_file_path, \
    ppmdecay_default_instructions_file_path, ppmdecay_default_results_file_path
from cmme.drex.worker import DREXModel
from cmme.ppmdecay.model import PPMModel, PPMDecayInstance
from cmme.visualization.data_frame import DataFrame


def test_init_succeeds():
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_sequence = [1, 1, 2, 3, 4, 4, 5, 6]

        ppm_instance = PPMDecayInstance()
        ppm_instance.input_sequence(input_sequence).alphabet_levels([1,2,3,4,5,6])
        ppm_model = PPMModel(ppm_instance)
        ppmdecay_instructions_file_path = ppmdecay_default_instructions_file_path(None, tmpdirname)
        ppmdecay_results_file_path = ppmdecay_default_results_file_path(None, tmpdirname)
        ppm_model.run(ppmdecay_instructions_file_path, ppmdecay_results_file_path)

        drex_prior = UnprocessedPrior(DistributionType.GAUSSIAN, [1, 1, 1.5, 2], 2)
        drex_instance = DREXInstructionBuilder()
        drex_instance.prior(drex_prior).input_sequence(input_sequence)
        drex_instructions_file_path = drex_default_instructions_file_path(None, tmpdirname)
        drex_results_file_path = drex_default_results_file_path(None, tmpdirname)
        drex_instance\
            .to_instructions_file()\
            .save_self(drex_instructions_file_path, drex_results_file_path)
        drex_model = DREXModel()
        drex_model.run(drex_instructions_file_path)

        data_frame = DataFrame(ppmdecay_results_file_path, drex_results_file_path, input_sequence)

        print(data_frame)
