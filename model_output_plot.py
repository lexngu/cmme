from .model_output_aggregator import *
from .ppm import PPMOutputParameters
from .drex import DREXOutputParameters
from .matlab_worker import *
import shutil
from pathlib import Path


class ModelOutputPlot:
    """This class generates the comparison plot."""

    def __init__(self, model_io_paths, ppm_output_path, drex_output_path):
        self._model_io_paths = model_io_paths

        ppm_output_parameters = PPMOutputParameters.from_csv(ppm_output_path)
        drex_output_parameters = DREXOutputParameters.from_mat(drex_output_path)
        self._aggregator = ModelOutputAggregator(ppm_output_parameters, drex_output_parameters)

        self._file_input_path = self._model_io_paths["input_file_path"] + ".mat"
        self._aggregator.write_mat(self._file_input_path)

        self._matlab_worker = MatlabWorker()

    def plot(self):
        result = self._matlab_worker.plot(self._file_input_path)
        result_figures = result['content']['figures']
        res = []
        for figure_path in result_figures:
            figure_name = Path(figure_path).name
            figure_destination_path = self._model_io_paths["generic_filename_prefix"] + figure_name
            shutil.copyfile(figure_path, figure_destination_path)
            res.append(figure_destination_path)
        return res
