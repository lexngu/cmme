from cmme.drex.base import DistributionType, UnprocessedPrior
from cmme.drex.model import DREXInstructionBuilder
from cmme.drex.worker import DREXModel
from cmme.ppmdecay.model import PPMModel, PPMDecayInstance
from cmme.visualization.data_frame import DataFrame
from cmme.visualization.plot import MatlabPlot, MatplotlibPlot


def test_matlab_plot():
    input_sequence = [1, 1, 2, 3, 4, 4, 5, 6]
    ppm_instance = PPMDecayInstance()
    ppm_instance.input_sequence(input_sequence).alphabet_levels([1, 2, 3, 4, 5, 6])
    ppm_model = PPMModel(ppm_instance)
    ppm_results_file = ppm_model.run()
    drex_prior = UnprocessedPrior(DistributionType.GAUSSIAN, [1, 1, 1.5, 2], 2)
    drex_instance = DREXInstructionBuilder()
    drex_instance.prior(drex_prior).input_sequence(input_sequence)
    instructions_file_path = drex_instance.build_instructions_file().write_to_mat()
    drex_model = DREXModel()
    drex_results_file = drex_model.run(instructions_file_path)
    data_frame = DataFrame(ppm_results_file, drex_results_file, input_sequence)
    matlab_plot = MatlabPlot(data_frame)

    results = matlab_plot.plot()
    assert len(results) >= 1

def test_matplotlib_plot():
    input_sequence = [1, 1, 2, 3, 4, 4, 5, 6]
    ppm_instance = PPMDecayInstance()
    ppm_instance.input_sequence(input_sequence).alphabet_levels([1, 2, 3, 4, 5, 6])
    ppm_model = PPMModel(ppm_instance)
    ppm_results_file = ppm_model.run()
    drex_prior = UnprocessedPrior(DistributionType.GAUSSIAN, [1, 1, 1.5, 2], 2)
    drex_instance = DREXInstructionBuilder()
    drex_instance.prior(drex_prior).input_sequence(input_sequence)
    instructions_file_path = drex_instance.build_instructions_file().write_to_mat()
    drex_model = DREXModel()
    drex_results_file = drex_model.run(instructions_file_path)
    data_frame = DataFrame(ppm_results_file, drex_results_file, input_sequence)
    matplotlib_plot = MatplotlibPlot(data_frame)

    results = matplotlib_plot.plot()
    assert results is not None