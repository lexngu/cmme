from __future__ import annotations

import numbers
from pathlib import Path
from typing import Union

import numpy as np
import scipy.io as sio

from .base import DistributionType, Prior, UnprocessedPrior, GaussianPrior, LognormalPrior, GmmPrior, PoissonPrior
from .util import trialtimefeature_sequence_as_multitrial_cell, \
    trialtimefeature_sequence_as_singletrial_array
from ..lib.input_data import auto_convert_input_sequence
from ..lib.instructions_file import InstructionsFile
from ..lib.results_file import ResultsFile


def to_mat(data, file_path: Union[str, Path]):
    sio.savemat(str(file_path), data)


def from_mat(file_path: Union[str, Path]) -> dict:
    mat_data = sio.loadmat(str(file_path))

    return mat_data


class DREXInstructionsFile(InstructionsFile):
    @staticmethod
    def _generate_instructions_file_path() -> Path:
        return Path("./drex-instructions.mat")

    @classmethod
    def _save(cls, instructions_file: DREXInstructionsFile, file_path: Union[str, Path]):
        data = dict()

        # Add instructions for procesing an unprocessed prior using D-REX's estimate_suffstat.m
        if isinstance(instructions_file.prior, UnprocessedPrior):
            data["estimate_suffstat"] = {
                "xs": trialtimefeature_sequence_as_multitrial_cell(instructions_file.prior.prior_input_sequence),
                "params": {
                    "distribution": instructions_file.prior.distribution_type().value,
                    "D": float(instructions_file.prior.D_value())
                }
            }
            if instructions_file.prior.distribution_type() == DistributionType.GMM:
                data["estimate_suffstat"]["params"]["max_ncomp"] = instructions_file.prior.max_n_comp

        # Add instructions for invoking D-REX (run_DREX_model.m)
        data["run_DREX_model"] = {
            "x": trialtimefeature_sequence_as_singletrial_array(instructions_file.input_sequence),
            "params": {
                "distribution": instructions_file.prior.distribution_type().value,
                "D": float(instructions_file.prior.D_value()),
                "hazard": float(instructions_file.hazard),
                "obsnz": instructions_file.obsnz,
                "memory": instructions_file.memory,
                "maxhyp": instructions_file.maxhyp,
                "predscale": instructions_file.predscale
            },
        }
        if instructions_file.prior.distribution_type() == DistributionType.GMM:
            data["run_DREX_model"]["params"]["max_ncomp"] = instructions_file.prior.max_n_comp

        # Add instructions for post_DREX_changedecision.m
        if instructions_file.change_decision_threshold is not None:
            data["post_DREX_changedecision"] = {
                "threshold": float(instructions_file.change_decision_threshold)
            }

        # Add results_file_path
        data["results_file_path"] = str(instructions_file.results_file_path)

        # Write and return
        return to_mat(data, file_path)

    @classmethod
    def save(cls, instructions_file: DREXInstructionsFile, file_path: Union[str, Path] = _generate_instructions_file_path()):
        cls._save(instructions_file, file_path)

    @staticmethod
    def load(file_path: str) -> InstructionsFile:
        pass

    def __init__(self, results_file_path: Union[str, Path], input_sequence: np.ndarray, prior: Prior,
                 hazard: Union[numbers.Number, list, np.ndarray], memory: Union[int, float], maxhyp: Union[int, float],
                 obsnz: float, predscale: float, change_decision_threshold: float = None):
        # TODO move predscale to last position
        super().__init__()
        input_sequence_length = len(input_sequence)
        if not isinstance(prior, Prior):
            raise ValueError("prior invalid! Should be an instance of drex.base.Prior.")
        if isinstance(hazard, numbers.Number):
            hazard_length = 1
        elif isinstance(hazard, list) or isinstance(hazard, np.ndarray):
            hazard_length = len(hazard)
        else:
            raise ValueError("hazard invalid! Should be scalar, np.ndarray, or list.")
        if hazard_length > 1:
            # If hazard is not scalar (i.e. more than one element), then there must be one value for each time step.
            if input_sequence_length != hazard_length:
                raise ValueError(
                    "Values invalid! If there is more than one hazard rate, "
                    "the number of hazard rates must equal the number of input sequence data.")
        if not (isinstance(memory, int) or memory == float('inf')):
            raise ValueError("memory invalid! Should be an integer or float('inf').")
        if not (isinstance(maxhyp, int) or maxhyp == float('inf')):
            raise ValueError("maxhyp invalid! Should be an integer or float('inf').")

        # convert to list of float(s)
        obsnz = [obsnz] if not isinstance(obsnz, list) else obsnz
        obsnz = [float(o) for o in obsnz]

        self.input_sequence = auto_convert_input_sequence(input_sequence)
        """Input sequence: time, feature => 1"""
        self.results_file_path = results_file_path
        """Where to store the model's results"""
        self.prior = prior
        """Prior distribution"""
        self.hazard = hazard
        """Hazard rate(s): scalar or list of numbers (for each input datum)"""
        self.obsnz = obsnz
        """Observation noise: feature => 1"""
        self.memory = memory
        """Number of most-recent hypotheses to calculate"""
        self.maxhyp = maxhyp
        """Number of hypotheses to calculate at every time step"""
        self.predscale = predscale
        """Predscale (D-REX internal)"""
        self.change_decision_threshold = change_decision_threshold
        """Threshold used for change detector"""


class ResultsFilePsi:
    def __init__(self, predictions: dict, positions: dict):
        """

        :param predictions: dictionary with key=feature, value=np.array of shape (time, position)
        :param positions: dictionary with key=feature, value=list with positions
        """
        predictions_features = list(predictions.keys())
        positions_features = list(positions.keys())
        # Check key set
        if set(predictions_features) != set(positions_features):  # use set to ignore ordering
            raise ValueError("predictions and positions invalid! Their key set must match.")

        # Check positions for each feature
        for f in predictions_features:
            predictions_positions = predictions[f].shape[1]
            positions_length = len(positions[f])

            if predictions_positions != positions_length:
                raise ValueError("predictions and positions invalid! The number of positions must match.")

        # Set attributes
        self._features = predictions_features
        self._feature_to_positions = positions
        self._feature_to_predictions = predictions

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


def parse_post_DREX_prediction_results(results):
    predictions = dict()
    positions = dict()

    nfeature = len(results)

    for f in range(nfeature):
        predictions[f] = np.array(results[f][0]["prediction"][0][0])  # convert to np.array with shape (time,position)

        f_positions = results[f][0]["positions"][0][0]
        if isinstance(f_positions, int) or isinstance(f_positions, float):  # convert
            positions[f] = np.array([f_positions])
        elif isinstance(f_positions, np.ndarray):
            positions[f] = np.array(f_positions)[0]

    return ResultsFilePsi(predictions, positions)


class DREXResultsFile(ResultsFile):
    @staticmethod
    def _generate_results_file_path() -> Path:
        return Path("drex-results.mat")

    @staticmethod
    def _save(results_file: DREXResultsFile, file_path: Union[str, Path]):
        raise NotImplementedError  # TODO

    @staticmethod
    def load(file_path: Union[str, Path]) -> Union[DREXResultsFile, Prior]:
        data = from_mat(file_path)

        prior = parse_results_file_estimate_suffstat(data)
        if "run_DREX_model_results" not in data:
            return prior

        instructions_file_path = data["instructions_file_path"][0]
        input_sequence = data["input_sequence"]
        run_results = data["run_DREX_model_results"]
        bd_results = data["post_DREX_beliefdynamics_results"]
        cd_results = data["post_DREX_changedecision_results"]
        pred_results = data["post_DREX_prediction_results"]

        input_sequence = trialtimefeature_sequence_as_singletrial_array(
            auto_convert_input_sequence([input_sequence.T.tolist()]))
        surprisal = np.array(run_results["surprisal"][0][0])
        joint_surprisal = np.array(run_results["joint_surprisal"][0][0]).flatten()
        context_beliefs = np.array(run_results["context_beliefs"][0][0])
        belief_dynamics = np.array(bd_results).flatten()
        change_decision_changepoint = float(cd_results["changepoint"][0][0])
        change_decision_probability = np.array(cd_results["changeprobability"][0][0]).flatten()
        change_decision_threshold = float(data["change_decision_threshold"])
        if prior.distribution_type() in [DistributionType.GAUSSIAN, DistributionType.LOGNORMAL, DistributionType.GMM]:
            psi = parse_post_DREX_prediction_results(pred_results)
        else:
            psi = ResultsFilePsi({}, {})

        return DREXResultsFile(str(file_path), str(instructions_file_path), input_sequence, prior, surprisal,
                               joint_surprisal,
                               context_beliefs, belief_dynamics, change_decision_changepoint,
                               change_decision_probability,
                               change_decision_threshold, psi)

    # TODO add prediction_params from run_DREX_model.m?
    def __init__(self, results_file_path: Union[str, Path], instructions_file_path: Union[str, Path],
                 input_sequence: np.ndarray, prior: Prior,
                 surprisal: np.ndarray, joint_surprisal: np.ndarray, context_beliefs: np.ndarray,
                 belief_dynamics: np.ndarray, change_decision_changepoint: int, change_decision_probability: np.ndarray,
                 change_decision_threshold: float, psi: ResultsFilePsi):
        super().__init__()
        if len(input_sequence.shape) != 2:
            raise ValueError("Shape of input_sequence invalid! Expected two dimensions: time, feature.")
        if not isinstance(prior, Prior):
            raise ValueError("Prior must be an instance of cmme.drex.base.Prior.")
        if len(surprisal.shape) != 2:
            raise ValueError("Shape of surprisal invalid! Expected two dimensions: time, feature.")
        if len(joint_surprisal.shape) != 1:
            raise ValueError("Shape of joint_surprisal invalid! Expected one dimension: time.")
        if len(context_beliefs.shape) != 2:
            raise ValueError("Shape of context_beliefs invalid! Expected two dimensions: time, context.")
        if len(belief_dynamics.shape) != 1:
            raise ValueError("Shape of belief_dynamics invalid! Expected one dimension: time.")

        [input_sequence_times, input_sequence_features] = input_sequence.shape
        [surprisal_times, surprisal_features] = surprisal.shape
        joint_surprisal_times = joint_surprisal.shape[0]
        [_, context_beliefs_contexts] = context_beliefs.shape
        belief_dynamics_times = belief_dynamics.shape[0]

        if not (
                input_sequence_times == surprisal_times and surprisal_times == joint_surprisal_times and joint_surprisal_times == (
                belief_dynamics_times - 1)):
            raise ValueError(
                "Dimension 'time' invalid! Value must be equal for input_sequence({}), surprisal({}), joint_surprisal({}), and belief_dynamics({}-1).".format(
                    input_sequence_times, surprisal_times, joint_surprisal_times, belief_dynamics_times))
        if not (input_sequence_features == surprisal_features):
            raise ValueError(
                "Dimension 'feature' invalid! Value must be equal for input_sequence, and surprisal.")

        self.dimension_values = dict()
        self.dimension_values["time"] = input_sequence_times
        self.dimension_values["feature"] = input_sequence_features
        self.dimension_values["context"] = context_beliefs_contexts

        self.results_file_path = results_file_path
        """File path to source data of this object"""
        self.instructions_file_path = instructions_file_path
        """Corresponding instructions file"""
        self.prior = prior
        """Prior"""
        self.input_sequence = input_sequence
        """Input sequence processed: time, feature => 1"""
        self.surprisal = surprisal
        """Surprisal values: time, feature => 1"""
        self.joint_surprisal = joint_surprisal
        """Joint surprisal values: time => 1"""
        self.context_beliefs = context_beliefs
        """Context belief distribution: memory, context => 1"""
        self.belief_dynamics = belief_dynamics
        """Belief dynamics: time => 1"""
        self.change_decision_changepoint = change_decision_changepoint
        """Change detector's calculated changepoint (although being an int, 
        this value is stored as float, in order to support float('nan'))"""
        self.change_decision_probability = change_decision_probability
        """Change detector's calculated change probability: time => 1"""
        self.change_decision_threshold = change_decision_threshold
        """Change detector's threshold: 1"""
        self.psi = psi
        """Marginal (predictive) probability distribution"""


def parse_results_file_estimate_suffstat(data) -> Prior:
    if "estimate_suffstat_results" not in data:
        raise ValueError("Missing dictionary key 'estimate_suffstat_results'!")
    if "distribution" not in data:
        raise ValueError("Missing dictionary key 'distribution'!")
    distribution = data["distribution"]
    es_data = data["estimate_suffstat_results"]

    if distribution == DistributionType.GAUSSIAN.value:
        data_means = es_data["mu"][0][0]
        data_covariance = es_data["ss"][0][0]
        data_n = es_data["n"][0][0]

        if data_means.shape[0] != data_covariance.shape[0] or data_covariance.shape[0] != data_n.shape[0]:
            raise ValueError(
                "estimate_suffstat_results invalid! The number of features between mu, ss, and/or n is not identical.")
        feature_count = data_means.shape[0]

        means = []
        covariance = []
        n = []
        for f_idx in range(feature_count):
            means.append(data_means[f_idx][0].flatten())  # 1d
            covariance.append(data_covariance[f_idx][0])  # 2d
            n.append(data_n[f_idx][0].flatten()[0])  # int

        return GaussianPrior(np.array(means), np.array(covariance), np.array(n))
    elif distribution == DistributionType.LOGNORMAL.value:
        data_means = es_data["mu"][0][0]
        data_covariance = es_data["ss"][0][0]
        data_n = es_data["n"][0][0]

        if data_means.shape[0] != data_covariance.shape[0] or data_covariance.shape[0] != data_n.shape[0]:
            raise ValueError(
                "estimate_suffstat_results invalid! The number of features between mu, ss, and/or n is not identical.")
        feature_count = data_means.shape[0]

        means = []
        covariance = []
        n = []
        for f_idx in range(feature_count):
            means.append(data_means[f_idx][0].flatten())  # 1d
            covariance.append(data_covariance[f_idx][0])  # 2d
            n.append(data_n[f_idx][0].flatten()[0])  # int

        return LognormalPrior(np.array(means), np.array(covariance), np.array(n))
    elif distribution == DistributionType.GMM.value:  # estimate_suffstat calculates a GMM with one single component
        data_means = es_data["mu"][0][0]
        data_covariance = es_data["sigma"][0][0]
        data_n = es_data["n"][0][0]
        data_pi = es_data["pi"][0][0].T
        data_sp = es_data["sp"][0][0].T
        data_k = es_data["k"][0][0].T

        if data_means.shape[0] != data_covariance.shape[0] or data_covariance.shape[0] != data_n.shape[0]:
            raise ValueError(
                "estimate_suffstat_results invalid! The number of features between mu, ss, and/or n is not identical.")
        feature_count = data_means.shape[0]

        means = []
        covariance = []
        n = []
        pi = []
        sp = []
        k = []
        for f_idx in range(feature_count):
            means.append(data_means[f_idx][0].flatten())  # 1d
            covariance.append(data_covariance[f_idx][0][0])  # 1d
            n.append(data_n[f_idx][0].flatten())  # 1d
            pi.append(data_pi[f_idx][0].flatten())  # 1d
            sp.append(data_sp[f_idx][0].flatten())  # 1d
            k.append(data_k[f_idx][0].flatten()[0])  # int

        return GmmPrior(np.array(means), np.array(covariance), np.array(n), np.array(pi), np.array(sp), np.array(k))
    elif distribution == DistributionType.POISSON.value:
        data_lambda = es_data["lambda"][0][0]
        data_n = es_data["n"][0][0]

        feature_count = data_lambda.shape[0]

        _lambd = []
        _n = []
        for f_idx in range(feature_count):
            _lambd.append(data_lambda[f_idx][0][0][0])
            _n.append(data_n[f_idx][0][0][0])

        return PoissonPrior(np.array(_lambd), np.array(_n))
