from .model_output_aggregator import *
from .ppm import PPMOutputParameters
from .drex import DREXOutputParameters
from .matlab_worker import *
import shutil
from pathlib import Path


class ModelOutputPlot:
    """This class generates the comparison plot."""

    def __init__(self, ppm_output_path: Path, drex_output_path: Path, plot_output_base_path: Path):
        self.plot_output_base_path = plot_output_base_path

        ppm_output_parameters = PPMOutputParameters.from_csv(str(ppm_output_path))
        drex_output_parameters = DREXOutputParameters.from_mat(str(drex_output_path))
        self._aggregator = ModelOutputAggregator(ppm_output_parameters, drex_output_parameters)

        self._plot_input_file_path = str(self.plot_output_base_path) + "-input.mat"
        self._aggregator.write_mat(self._plot_input_file_path)

        self._matlab_worker = MatlabWorker()

    def plot(self):
        result = self._matlab_worker.plot(self._plot_input_file_path)
        result_figures = result['content']['figures']
        res = []
        for figure_path in result_figures:
            figure_name = Path(figure_path).name
            figure_destination_path = str(self.plot_output_base_path.parent / (str(self.plot_output_base_path.stem) + "-" + figure_name))
            shutil.copyfile(figure_path, figure_destination_path)
            res.append(figure_destination_path)
        return res
