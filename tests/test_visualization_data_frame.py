from cmme.drex.distribution.base import DistributionType
from cmme.drex.distribution.prior import UnprocessedPrior
from cmme.drex.model import DREXModel
from cmme.ppmdecay.instance import ModelType
from cmme.ppmdecay.model import PPMModel
from cmme.visualization.data_frame import DataFrame


def test_init_succeeds():
    input_sequence = [1, 1, 2, 3, 4, 4, 5, 6]
    ppm_model = PPMModel(ModelType.DECAY)
    ppm_model.instance.input_sequence(input_sequence).alphabet_levels([1,2,3,4,5,6])
    ppm_results_file = ppm_model.run()
    drex_prior = UnprocessedPrior(DistributionType.GAUSSIAN, [1, 1, 1.5, 2], 2)
    drex_model = DREXModel(drex_prior)
    drex_model.instance.with_input_sequence(input_sequence)
    drex_results_file = drex_model.run()
    data_frame = DataFrame(ppm_results_file, drex_results_file, input_sequence)

    print(data_frame)
