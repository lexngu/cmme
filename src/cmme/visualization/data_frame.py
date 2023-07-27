import numpy as np
import pandas as pd
import scipy.io as sio
from cmme.drex.binding import DREXResultsFile
from cmme.ppmdecay.binding import PPMResultsMetaFile
from cmme.visualization.util.util import cmme_default_plot_instructions_file_path


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
        ppm_alphabet_size = len(self.ppm_results_file._alphabet_levels)
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

    def write_to_mat(self, instructions_file_path = cmme_default_plot_instructions_file_path()):
        data = {
            "ppm_results_file_path": str(self.ppm_results_file.results_file_meta_path),
            "drex_results_file_path": str(self.drex_results_file.results_file_path),
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