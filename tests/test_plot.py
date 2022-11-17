from cmme.drex.distributions import UnprocessedDrexDistributionContainer, DistributionType
from cmme.drex.model import DREXModel
from cmme.ppmdecay.instance import ModelType
from cmme.ppmdecay.model import PPMModel
from cmme.visualization.data_frame import DataFrame
from cmme.visualization.plot import MatlabPlot


def test_matlab_plot():
    input_sequence = [1, 1, 2, 3, 4, 4, 5, 6]
    ppm_model = PPMModel(ModelType.DECAY)
    ppm_model.instance.input_sequence(input_sequence).alphabet_levels([1, 2, 3, 4, 5, 6])
    ppm_results_file = ppm_model.run()
    drex_prior = UnprocessedDrexDistributionContainer(DistributionType.GAUSSIAN, [1, 1, 1.5, 2], 2)
    drex_model = DREXModel(drex_prior)
    drex_model.instance.with_input_sequence(input_sequence)
    drex_results_file = drex_model.run()
    data_frame = DataFrame(ppm_results_file, drex_results_file, input_sequence)
    matlab_plot = MatlabPlot(data_frame)

    results = matlab_plot.plot()
    assert len(results) >= 1

def test_matplotlib_plot():
    pass