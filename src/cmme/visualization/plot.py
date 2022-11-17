import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from cmme.drex.util.matlab import MatlabWorker
from cmme.visualization.data_frame import DataFrame
import numpy as np

from cmme.visualization.util.util import cmme_default_plot_output_file_path


class Plot(ABC):
    def __init__(self, data_frame: DataFrame):
        self.data_frame = data_frame

    @abstractmethod
    def plot(self):
        pass

class MatlabPlot(Plot):
    def __init__(self, data_frame: DataFrame):
        super().__init__(data_frame)

    def plot(self, plot_output_file_path = cmme_default_plot_output_file_path()):
        data_frame_path = self.data_frame.write_to_mat()
        result = MatlabWorker.plot(data_frame_path)
        result_figures = result['content']['figures']
        res = []
        for figure_path in result_figures:
            figure_name = Path(figure_path).name
            figure_destination_path = str(
                plot_output_file_path.parent / (str(plot_output_file_path.stem) + "-" + figure_name))
            shutil.copyfile(figure_path, figure_destination_path)
            res.append(figure_destination_path)
        return res

class MatplotlibPlot(Plot):
    def __init__(self, data_frame: DataFrame):
        super().__init__(data_frame)

    def plot(self) -> Figure:
        df = self._aggregator.df

        # TODO remove hard coded removing of last row
        df = df.drop(df.tail(1).index)

        df_as_dict = {name: col.values for name, col in df.items()}
        ppm_alphabet_size = int(df["ppm_alphabet_size"][0])
        ntime = len(df["observation"])

        plt.rcParams['image.cmap'] = 'viridis'
        plt.rcParams['font.family'] = 'Helvetica, Arial'

        maxSubplot = 11
        xlims = (0, len(df_as_dict["observation"]) - 1)

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
        x = list(range(1, ntime + 1))
        y = list(range(1, ppm_alphabet_size + 1))
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

