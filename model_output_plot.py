from .model_output_aggregator import *
from .ppm import PPMOutputParameters
from .drex import DREXOutputParameters
from .matlab_worker import *
import shutil
from pathlib import Path

from matplotlib import pyplot as plt 
import numpy as np


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

    def _prepare_sequence(self, sequence):
        xpos = []
        ypos = []
        for idx,e in enumerate(sequence):
            xpos.extend([idx, idx+.15/.175, None])
        for idx,e in enumerate(sequence):
            ypos.extend([e, e, None])

        return [xpos, ypos]

    def _prepare_predictions(self, predictions):
        result = np.array([])
        size = 0
        for e in predictions:
            size = len(e)
            result = np.concatenate((result, e))
        result = result.reshape(int(len(result)/size), size)
        return result

    def _prepare_context_beliefs(self, context_beliefs):
        result = np.array([])
        size = 0
        for e in context_beliefs:
            size = len(e)
            result = np.concatenate((result, e))
        result = result.reshape(int(len(result)/size), size)
        return result

    def plot_matplotlib(self):
        """Returns a matplotlib figure"""
        df = self._aggregator.df

        # TODO remove hard coded removing of last row
        df = df.drop(df.tail(1).index)

        df_as_dict = {name: col.values for name, col in df.items()}
        ppm_alphabet_size = int(df["ppm_alphabet_size"][0])
        ntime = len(df["observation"])

        plt.rcParams['image.cmap'] = 'viridis'
        plt.rcParams['font.family'] = 'Helvetica, Arial'

        maxSubplot = 11
        xlims = (0,len(df_as_dict["observation"])-1)

        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(50)

        ax = plt.subplot(maxSubplot, 1, 1)
        data = df_as_dict["observation"]
        [xpos, ypos] = self._prepare_sequence(data)
        ax.plot(xpos, ypos, linewidth=3)
        ax.set_title("Input Sequence")
        ax.set_xlabel("Time")
        ax.set_ylabel("Observation")
        ax.set_xlim(xlims)

        ax = plt.subplot(maxSubplot, 1, 2)
        ax.set_title("PPM: Predictions")
        data = df_as_dict["ppm_probability_distribution"]
        predictions = self._prepare_predictions(data).T
        x = list(range(1, ntime+1))
        y = list(range(1, ppm_alphabet_size+1))
        ax.contourf(x, y, predictions, 100)
        ax.set_ylabel("Observation")
        ax.set_xlim(xlims)
        ax.yaxis.get_major_locator().set_params(integer=True)

        ax = plt.subplot(maxSubplot, 1, 3)
        ax.set_title("D-REX: Predictions")
        data = df_as_dict["drex_predictions"]
        predictions = self._prepare_predictions(data).T
        x = list(range(1, ntime + 1))
        y = list(range(1, ppm_alphabet_size + 1))
        ax.contourf(x, y, predictions, 1000)
        ax.set_ylabel("Observation")
        ax.set_xlim(xlims)
        ax.yaxis.get_major_locator().set_params(integer=True)

        ax = plt.subplot(maxSubplot, 1, 4)
        ax.set_title("PPM: Model Order")
        x = list(range(1, ntime + 1))
        y = df_as_dict["ppm_model_order"]
        ax.plot(x, y)
        ax.set_ylabel("Model Order")
        ax.set_xlim(xlims)
        ax.yaxis.get_major_locator().set_params(integer=True)

        max_ppm_drex_ic = max(np.amax(df_as_dict["ppm_information_content"]), np.amax(df_as_dict["drex_surprisal"]))

        ax = plt.subplot(maxSubplot, 1, 5)
        ax.set_title("PPM: Information Content")
        data = df_as_dict["ppm_information_content"]
        ax.plot(data)
        ax.set_xlim(xlims)
        ax.set_ylim((0, max_ppm_drex_ic))

        ax = plt.subplot(maxSubplot, 1, 6)
        ax.set_title("D-REX: Surprisal")
        data = df_as_dict["drex_surprisal"]
        ax.plot(data)
        ax.set_xlim(xlims)
        ax.set_ylim((0, max_ppm_drex_ic))

        ax = plt.subplot(maxSubplot, 1, 7)
        ax.set_title("PPM: Entropy")
        data = df_as_dict["ppm_entropy"]
        ax.plot(data)
        ax.set_xlim(xlims)
        ax.set_ylim(bottom=0)

        ax = plt.subplot(maxSubplot, 1, 8)
        ax.set_title("D-REX: Entropy")
        data = df_as_dict["drex_entropy"]
        ax.plot(data)
        ax.set_xlim(xlims)
        ax.set_ylim(bottom=0)

        ax = plt.subplot(maxSubplot, 1, 9)
        ax.set_title("D-REX: Context Beliefs")
        data = df_as_dict["drex_context_beliefs"]
        context_beliefs = self._prepare_context_beliefs(data).T
        context_beliefs[context_beliefs == 0] = np.nan
        p = plt.pcolor(np.log10(context_beliefs))
        ax.set_xlim(xlims)

        ax = plt.subplot(maxSubplot, 1, 10)
        ax.set_title("D-REX: Belief Dynamics")
        data = df_as_dict["drex_bd"]
        ax.plot(data)
        ax.set_xlim(xlims)
        ax.set_ylim((0, 1))

        ax = plt.subplot(maxSubplot, 1, 11)
        ax.set_title("D-REX: Change Probability")
        data = df_as_dict["drex_cd_probability"]
        ax.plot(data)
        ax.set_xlim(xlims)
        ax.set_ylim((0, 1))

        if df_as_dict["drex_cd_changepoint"].shape[0] > 0:
            changepoint = df_as_dict["drex_cd_changepoint"].any()
            ax.axvline(changepoint, color='grey')
            plt.text(changepoint, 0.5, "Changepoint " + str(changepoint), rotation=90)
            
        return f
