from cmme.drex.base import DistributionType, UnprocessedPrior
from cmme.drex.model import DREXModel, DREXInstructionBuilder
from cmme.ppmdecay.base import ModelType
from cmme.ppmdecay.model import PPMModel, PPMDecayInstance
from cmme.visualization.data_frame import DataFrame


def test_init_succeeds():
    input_sequence = [1, 1, 2, 3, 4, 4, 5, 6]
    ppm_instance = PPMDecayInstance()
    ppm_instance.input_sequence(input_sequence).alphabet_levels([1,2,3,4,5,6])
    ppm_model = PPMModel(ppm_instance)
    ppm_results_file = ppm_model.run()
    drex_prior = UnprocessedPrior(DistributionType.GAUSSIAN, [1, 1, 1.5, 2], 2)
    drex_instance = DREXInstructionBuilder()
    drex_instance.prior(drex_prior).input_sequence(input_sequence)
    drex_model = DREXModel(drex_instance)
    drex_results_file = drex_model.run()
    data_frame = DataFrame(ppm_results_file, drex_results_file, input_sequence)

    print(data_frame)
