import tempfile

import numpy as np
import pandas as pd
import scipy.io as sio
from cmme.drex.binding import DREXResultsFile
from cmme.ppmdecay.binding import PPMResultsMetaFile

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from cmme.drex.worker import MatlabWorker


class DataFrame:
    """Aggregates ResultsFiles into a single dataframe (if the contained input sequence is commensurable)"""

    def __init__(self, ppm_results_file: PPMResultsMetaFile, drex_results_file: DREXResultsFile, input_sequence):
        """
        :param ppm_results_file:
        :param drex_results_file:
        :param input_sequence: Input sequence to use (in D-REX: the feature index is automatically detected)
        """
        self.ppm_results_file = ppm_results_file
        self.drex_results_file = drex_results_file
        self.drex_feature_index = None

        self.df = self._build_data_frame()

    def _build_data_frame(self) -> pd.DataFrame:
        ppm_data = self.ppm_results_file.results_file_data
        drex_data = self.drex_results_file

        ppm_input_sequence = ppm_data.df_of_last_trial()["symbol"].tolist()
        try:
            ppm_input_sequence_as_numbers = list(map(float, ppm_input_sequence))
        except:
            raise ValueError("PPM's input sequence was expected to but cannot be converted to a list of numbers.")

        drex_input_sequence = None
        drex_input_sequences = drex_data.input_sequence
        drex_input_sequence_feature_count = drex_data.dimension_values["feature"]
        for feature_index in range(drex_input_sequence_feature_count):
            drex_feature_input_sequence = drex_input_sequences[:, feature_index].tolist()
            input_sequences_are_equal = np.array_equal(drex_feature_input_sequence, ppm_input_sequence_as_numbers)
            if input_sequences_are_equal:
                self.drex_feature_index = feature_index
                drex_input_sequence = drex_feature_input_sequence
                break
        if drex_input_sequence is None:
            raise ValueError("Could not find a matching input sequence in any of D-REX's feature-specific input sequences.")

        # Data frame columns
        observations = drex_input_sequence
        ppm_information_content = ppm_data.df_of_last_trial()["information_content"]
        drex_joint_surprisal = drex_data.joint_surprisal
        ppm_entropy = ppm_data.df_of_last_trial()["entropy"]
        ppm_predictions = ppm_data.df_of_last_trial()["distribution"]
        drex_predictions = drex_data.psi.prediction_by_feature(self.drex_feature_index)
        ppm_model_order = ppm_data.df_of_last_trial()["model_order"]
        drex_context_beliefs = drex_data.context_beliefs
        drex_belief_dynamics = drex_data.belief_dynamics
        drex_changedecision_probability = drex_data.change_decision_probability
        # Constants
        ppm_alphabet_size = len(self.ppm_results_file.alphabet_levels)
        drex_positions = self.drex_results_file.psi.positions_by_feature(self.drex_feature_index)
        drex_changedecision_threshold = drex_data.change_decision_threshold  # TODO
        drex_changedecision_changepoint = drex_data.change_decision_changepoint

        data_frame_columns = ["observation",
                              "ppm_information_content", "drex_surprisal",
                              "ppm_alphabet_size", "ppm_model_order", "ppm_predictions", "drex_predictions",
                              "ppm_entropy", "drex_entropy",
                              "drex_context_beliefs",
                              "drex_bd",
                              "drex_cd_probability", "drex_cd_changepoint", "drex_cd_threshold"]
        data_frame = pd.DataFrame([], columns=data_frame_columns)

        # init data frame
        for idx, observation in enumerate(observations):
            drex_entropy = calc_drex_entropy(drex_predictions[idx])
            data_frame.loc[idx] = [observation,
                                   ppm_information_content[idx], drex_joint_surprisal[idx],
                                   ppm_alphabet_size, ppm_model_order[idx], ppm_predictions[idx], drex_predictions[idx],
                                   ppm_entropy[idx], drex_entropy,
                                   drex_context_beliefs[idx],
                                   drex_belief_dynamics[idx],
                                   drex_changedecision_probability[idx], drex_changedecision_changepoint,
                                   drex_changedecision_threshold]

        return data_frame

    def write_to_mat(self, instructions_file_path):
        data = {
            "data_frame": {name: col.values for name, col in self.df.items()}
        }

        # remove duplicates
        data["data_frame"]["ppm_alphabet_size"] = data["data_frame"]["ppm_alphabet_size"][0]
        data["data_frame"]["drex_cd_changepoint"] = data["data_frame"]["drex_cd_changepoint"][0]
        data["data_frame"]["drex_cd_threshold"] = data["data_frame"]["drex_cd_threshold"][0]

        sio.savemat(str(instructions_file_path), data) # Note: uses scipy.io because of easier use with np.arrays
        return instructions_file_path


def calc_drex_entropy(ensemble) -> float:
    ensemble_sum = np.sum(ensemble)
    entropy = 0
    for e in ensemble:
        e = e / ensemble_sum # normalization
        entropy += e * np.log2(e)
    entropy = -entropy
    return entropy


class Plot(ABC):
    def __init__(self, data_frame: DataFrame):
        self.data_frame = data_frame

    @abstractmethod
    def plot(self):
        pass


class MatlabPlot(Plot):
    def __init__(self, data_frame: DataFrame):
        super().__init__(data_frame)

    def plot(self, plot_output_file_path=None, instructions_file_path=None):
        """
        Write the instructions file and create the plot.

        Parameters
        ----------
        plot_output_file_path
            Path where the resulting plot is to be saved to
        instructions_file_path
            Path where the instructions file is to be saved to
        Returns
        -------
        res
            List of file paths
        """
        if instructions_file_path is None:
            instructions_file_path = Path(tempfile.NamedTemporaryFile().name)
            print("Instructions file path set to {}".format(instructions_file_path))
        if plot_output_file_path is None:
            plot_output_file_path = Path(tempfile.NamedTemporaryFile().name)
            print("Plot output file path set to {}".format(plot_output_file_path))

        data_frame_path = self.data_frame.write_to_mat(instructions_file_path)
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
        df = self.data_frame.df

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
        data = df_as_dict["ppm_predictions"]
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
        context_beliefs = self._prepare_context_beliefs(data)
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
