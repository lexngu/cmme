import matlab
import numpy as np

from cmme.drex.util.matlab import MatlabWorker


class ResultsFilePsi:
    def __init__(self, predictions, positions):
        """

        :param predictions: dictionary with key=feature, value=np.array of shape (time, position)
        :param positions: dictionary with key=feature, value=list with positions
        """
        predictions_features = list(predictions.keys())
        positions_features = list(positions.keys())
        # Check key set
        if set(predictions_features) != set(positions_features): # use set to ignore ordering
            raise ValueError("predictions and positions invalid! Their key set must match.")

        # Check positions for each feature
        for f in predictions_features:
            predictions_positions = predictions[f].shape[1]
            positions_length = len(positions[f])

            if predictions_positions != positions_length:
                raise ValueError("predictions and positions invalid! The number of positions must match.")

        # Check time for each feature
        time = predictions[0].shape[0]
        for f in predictions_features:
            if time != predictions[f].shape[0]:
                raise ValueError("predictions invalid! Each feature must have an equal number of time steps.")
            time = predictions[f].shape[0]

        # Set attributes
        self._features = predictions_features
        self._feature_to_positions = positions
        self._feature_to_predictions = predictions
        self._time = time

    def features(self):
        """
        :return: list of features
        """
        return self._features

    def prediction_by_feature(self, feature):
        """

        :param feature:
        :return: np.array of shape (time,position)
        """
        return self._feature_to_predictions[feature]

    def positions_by_feature(self, feature):
        """

        :param feature:
        :return: list with positions
        """
        return self._feature_to_positions[feature]

    def time(self):
        """

        :return: number of time steps
        """
        return self._time


def parse_post_DREX_prediction_results(results):
    predictions = dict()
    positions = dict()

    nfeature = len(results)

    for f in range(nfeature):
        predictions[f] = np.array(results[f]["prediction"], dtype=float) # convert to np.array with shape (time,position)

        f_positions = results[f]["positions"]
        if isinstance(f_positions, int) or isinstance(f_positions, float): # convert
            positions[f] = np.array([f_positions])
        elif isinstance(f_positions, matlab.double):
            positions[f] = np.array(f_positions)[0]

    return ResultsFilePsi(predictions, positions)


class ResultsFile:
    # TODO add prediction_params from run_DREX_model.m?
    def __init__(self, results_file_path, instructions_file_path, input_sequence, surprisal, joint_surprisal, context_beliefs,
                 belief_dynamics, change_decision_changepoint, change_decision_probability, change_decision_threshold, psi: ResultsFilePsi):
        if len(input_sequence.shape) != 2:
            raise ValueError("Shape of input_sequence invalid! Expected two dimensions: time, feature.")
        if len(surprisal.shape) != 2:
            raise ValueError("Shape of surprisal invalid! Expected two dimensions: time, feature.")
        if len(joint_surprisal.shape) != 2:
            raise ValueError("Shape of joint_surprisal invalid! Expected two dimensions: time, 1.")
        if len(surprisal.shape) != 2:
            raise ValueError("Shape of context_beliefs invalid! Expected two dimensions: time, context.")
        if len(belief_dynamics.shape) != 2:
            raise ValueError("Shape of belief_dynamics invalid! Expected one dimension: time, 1.")
        # TODO replace
        #if len(psi.shape) != 3:
        #    raise ValueError("Shape of psi invalid! Expected three dimensions: time, feature, position.")

        [input_sequence_times, input_sequence_features] = input_sequence.shape
        [surprisal_times, surprisal_features] = surprisal.shape
        joint_surprisal_times = joint_surprisal.shape[0]
        [context_beliefs_times, context_beliefs_contexts] = context_beliefs.shape
        belief_dynamics_times = belief_dynamics.shape[0]
        psi_features = len(psi.features())
        psi_times = psi.time()

        if not (input_sequence_times == surprisal_times and surprisal_times == joint_surprisal_times and joint_surprisal_times == (context_beliefs_times-1) and
                (context_beliefs_times-1) == (belief_dynamics_times-1) and (belief_dynamics_times-1) == psi_times):
            raise ValueError("Dimension 'time' invalid! Value must be equal for input_sequence({}), surprisal({}), joint_surprisal({}), context_beliefs({}), belief_dynamics({}), and psi({}).".format(input_sequence_times, surprisal_times, joint_surprisal_times, context_beliefs_times, belief_dynamics_times, psi_times))
        if not (input_sequence_features == surprisal_features and surprisal_features == psi_features):
            raise ValueError(
                "Dimension 'feature' invalid! Value must be equal for input_sequence, surprisal, and psi.")

        self.dimension_values = dict()
        self.dimension_values["time"] = input_sequence_times
        self.dimension_values["feature"] = input_sequence_features
        self.dimension_values["context"] = context_beliefs_contexts

        self.results_file_path = results_file_path
        """File path to source data of this object"""
        self.instructions_file_path = instructions_file_path
        """Corresponding instructions file"""
        self.input_sequence = input_sequence
        """Input sequence processed: time, feature => 1"""
        self.surprisal = surprisal
        """Surprisal values: time, feature => 1"""
        self.joint_surprisal = joint_surprisal
        """Joint surprisal values: time => 1"""
        self.context_beliefs = context_beliefs
        """Context belief distribution: time, context => 1"""
        self.belief_dynamics = belief_dynamics
        """Belief dynamics: time => 1"""
        self.change_decision_changepoint = change_decision_changepoint
        """Change detector's calculated changepoint"""
        self.change_decision_probability = change_decision_probability
        """Change detector's calculated change probability: time => 1"""
        self.change_decision_threshold = change_decision_threshold
        """Change detector's threshold: 1"""
        self.psi = psi
        """Marginal (predictive) probability distribution"""

def parse_results_file(results_file_path) -> ResultsFile:
    data = MatlabWorker.from_mat(results_file_path)

    instructions_file_path = data["instructions_file_path"]
    input_sequence = data["input_sequence"]
    run_results = data["run_DREX_model_results"]
    bd_results = data["post_DREX_beliefdynamics_results"]
    cd_results = data["post_DREX_changedecision_results"]
    pred_results = data["post_DREX_prediction_results"]

    input_sequence = np.array(input_sequence)
    surprisal = np.array(run_results["surprisal"])
    joint_surprisal = np.array(run_results["joint_surprisal"])
    context_beliefs = np.array(run_results["context_beliefs"])
    belief_dynamics = np.array(bd_results)
    change_decision_changepoint = cd_results["changepoint"]
    change_decision_probability = np.array(cd_results["changeprobability"])
    change_decision_threshold = float(data["change_decision_threshold"])
    psi = parse_post_DREX_prediction_results(pred_results)

    return ResultsFile(results_file_path, instructions_file_path, input_sequence, surprisal, joint_surprisal, context_beliefs, belief_dynamics, change_decision_changepoint, change_decision_probability, change_decision_threshold, psi)