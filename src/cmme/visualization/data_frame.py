import drex.matlab
import ppmdecay.r
import drex.results_file
import ppmdecay.results_file
import pandas as pd

import scipy.io as sio

class DataFrame:
    def __init__(self, ppm_results_file_path = None, drex_results_file_path = None):
        self._ppm_results_file_path = ppm_results_file_path
        self._drex_results_file_path = drex_results_file_path
        self._ppm_results_file: ppmdecay.results_file.ResultsMetaFile = None
        self._drex_results_file: drex.results_file.ResultsMetaFile = None

        if ppm_results_file_path is not None:
            self._ppm_results_file = ppmdecay.r.parse_results_meta_file(ppm_results_file_path)
        if drex_results_file_path is not None:
            self._drex_results_file = drex.matlab.parse_results_meta_file(drex_results_file_path)

        self.data_frame = self._build_data_frame()

    def _build_data_frame(self):
        prfdata = self._ppm_results_file.results_file_data
        drf = self._drex_results_file

        ppm_input_sequence = prfdata.symbols
        drex_feature = 0
        drex_input_sequence = drf.input_sequence.T[drex_feature] # TODO support multi-feature
        if len(ppm_input_sequence) != len(drex_input_sequence):
            raise ValueError("Results files invalid! The length of their input must match")

        # Column values: Time-variant
        observations = drex_input_sequence
        ppm_information_content = prfdata.information_contents
        drex_joint_surprisal = drf.joint_surprisal
        ppm_entropy = prfdata.entropies
        drex_entropy = [pd.NA]*len(ppm_input_sequence) # TODO how to correctly calculate entropy?
        ppm_predictions = prfdata.distributions
        drex_predictions = drf.psi.prediction_by_feature(drex_feature)
        ppm_model_order = prfdata.model_orders
        drex_context_beliefs = drf.context_beliefs
        drex_belief_dynamics = drf.belief_dynamics
        drex_changedecision_probability = drf.change_decision_probability
        # Constants
        ppm_alphabet_size = len(self._ppm_results_file._alphabet_levels)
        drex_positions = self._drex_results_file.psi.positions_by_feature(drex_feature)
        drex_changedecision_threshold = 0 # TODO
        drex_changedecision_changepoint = drf.change_decision_changepoint

        data_frame_columns = ["observation",
                            "ppm_information_content", "drex_surprisal",
                            "ppm_alphabet_size", "ppm_model_order", "ppm_predictions", "drex_predictions",
                            "ppm_entropy", "drex_entropy",
                            "drex_context_beliefs",
                            "drex_bd",
                            "drex_cd_probability", "drex_cd_changepoint", "drex_cd_threshold"]
        data_frame = pd.DataFrame([], columns = data_frame_columns)

        # init data frame
        for idx, observation in enumerate(observations):
            data_frame.loc[idx] = [observation,
                                   ppm_information_content[idx], drex_joint_surprisal[idx],
                                   ppm_alphabet_size, ppm_model_order[idx], ppm_predictions[idx], drex_predictions[idx],
                                   ppm_entropy[idx], drex_entropy[idx],
                                   drex_context_beliefs[idx],
                                   drex_belief_dynamics[idx],
                                   drex_changedecision_probability[idx], drex_changedecision_changepoint, drex_changedecision_threshold]

        return data_frame

def write_data_frame(data_frame: DataFrame, output_file_path = "data_frame.mat"):
    data = {
        "ppm_results_file_path": data_frame._ppm_results_file_path,
        "drex_results_file_path": data_frame._drex_results_file_path,
        "data_frame": {name: col.values for name, col in data_frame.data_frame.items()}
    }
    sio.savemat(output_file_path, data) # uses scipy.io because of easier use with np.arrays
    return output_file_path