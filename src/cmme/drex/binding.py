from __future__ import annotations

import numbers
from pathlib import Path
from typing import Union

import numpy as np
import scipy.io as sio

from .base import DistributionType, Prior, UnprocessedPrior, GaussianPrior, LognormalPrior, GmmPrior, PoissonPrior
from .util import transform_to_unified_drex_input_sequence_representation
from ..lib.util import nparray_to_list
from ..lib.instructions_file import InstructionsFile
from ..lib.results_file import ResultsFile


def to_mat(data: dict, file_path: Union[str, Path]):
    """
    Write data to a MATLAB file (.mat)

    Parameters
    ----------
    data
        Data to write
    file_path
        Path to write the file to
    """
    sio.savemat(str(file_path), data)


def from_mat(file_path: Union[str, Path], simplify_cells=True) -> dict:
    """
    Return the content of a MATLAB file (.mat)

    Parameters
    ----------
    file_path
        Path where the file is stored
    simplify_cells
        Wheter to use sio.loadmat(...)'s simplify_cells feature
    Returns
    -------
    dict
        Content of the MATLAB file
    """
    mat_data = sio.loadmat(str(file_path), simplify_cells=simplify_cells)

    return mat_data


def transform_to_estimatesuffstat_representation(input_sequence: np.ndarray) -> np.ndarray:
    """
    Transform an input sequence to a numpy array representation which suits the representation as needed by
    estimate_suffstat.m, i.e., the representation is of a multi-trial, multi-feature sequence.

    Parameters
    ----------
    input_sequence
        Unified representation of the input sequence, as resulting of
        cmme.drex.util.transform_to_unified_drex_input_sequence_representation(...)

    Returns
    -------
    np.ndarray
        Numpy array representation, shape: (trial, time, feature)
    """
    if input_sequence.dtype == object:
        res = np.empty((input_sequence.shape[0],), dtype=object)
        for idx, e in enumerate(input_sequence):
            res[idx] = e.tolist()
        return res
    else:
        raise ValueError("input_sequence should be a np.array(dtype=object) of three dimensions.")


def transform_to_rundrexmodel_representation(input_sequence: np.ndarray) -> np.ndarray:
    """
        Transform an input sequence to a numpy array representation which suits the representation as needed by
        run_DREX_model.m, i.e., the representation is of a single-trial, multi-feature sequence.

        Parameters
        ----------
        input_sequence
            Unified representation of the input sequence, as resulting of
            cmme.drex.util.transform_to_unified_drex_input_sequence_representation(...)

        Returns
        -------
        np.ndarray
            Numpy array representation, shape: (time, feature)
        """
    if input_sequence.dtype == object and input_sequence.shape[0] == 1:
        return np.array(input_sequence[0].tolist(), dtype=float)
    else:
        raise ValueError("input_sequence should be a np.array(dtype=object) of three dimensions, "
                         "and with a single trial.")


class DREXInstructionsFile(InstructionsFile):
    @staticmethod
    def save(instructions_file: DREXInstructionsFile, instructions_file_path: Union[str, Path],
             results_file_path: Union[str, Path] = None):
        data = dict()

        # Add instructions for procesing an unprocessed prior using D-REX's estimate_suffstat.m
        if isinstance(instructions_file.prior, UnprocessedPrior):
            data["estimate_suffstat"] = {
                "xs": transform_to_estimatesuffstat_representation(instructions_file.prior.prior_input_sequence),
                "params": {
                    "distribution": instructions_file.prior.distribution_type().value,
                    "D": float(instructions_file.prior.D_value())
                }
            }

        # Add instructions for invoking D-REX (run_DREX_model.m)
        data["run_DREX_model"] = {
            "x": transform_to_rundrexmodel_representation(instructions_file.input_sequence),
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
            data["run_DREX_model"]["params"]["max_ncomp"] = instructions_file.max_ncomp
            data["run_DREX_model"]["params"]["beta"] = instructions_file.beta

        # Add instructions for post_DREX_changedecision.m
        if instructions_file.change_decision_threshold is not None:
            data["post_DREX_changedecision"] = {
                "threshold": float(instructions_file.change_decision_threshold)
            }

        # Add results_file_path
        data["results_file_path"] = str(results_file_path) if results_file_path is not None else ""

        # Write
        to_mat(data, instructions_file_path)

    @staticmethod
    def load(file_path: Union[str, Path]) -> DREXInstructionsFile:
        data = from_mat(file_path)

        input_sequence = nparray_to_list(data["run_DREX_model"]["x"])
        data_rundrexmodel_params = data["run_DREX_model"]["params"]
        distribution = DistributionType(data_rundrexmodel_params["distribution"])
        hazard = data_rundrexmodel_params["hazard"]
        obsnz = data_rundrexmodel_params["obsnz"]
        memory = data_rundrexmodel_params["memory"]
        maxhyp = data_rundrexmodel_params["maxhyp"]
        predscale = data_rundrexmodel_params["predscale"]

        beta = None
        max_ncomp = None
        if distribution == DistributionType.GMM:
            beta = data_rundrexmodel_params["beta"]
            max_ncomp = data_rundrexmodel_params["max_ncomp"]

        change_decision_threshold = None
        if "post_DREX_changedecision" in data and "threshold" in data["post_DREX_changedecision"]:
            change_decision_threshold = data["post_DREX_changedecision"]["threshold"]

        if "estimate_suffstat" in data:
            prior_input_sequence = nparray_to_list(data["estimate_suffstat"]["xs"])
            data_estimatesuffstat_params = data["estimate_suffstat"]["params"]
            prior_distribution = DistributionType(data_estimatesuffstat_params["distribution"])
            prior_D = data_estimatesuffstat_params["D"]

            prior = UnprocessedPrior(prior_distribution, prior_input_sequence, prior_D)
        else:
            raise NotImplementedError("Processed priors not implemented yet.")

        return DREXInstructionsFile(input_sequence, prior,
                                    hazard, memory, maxhyp, obsnz,
                                    max_ncomp, beta,
                                    predscale, change_decision_threshold)

    def __init__(self, input_sequence: np.ndarray, prior: Prior,
                 hazard: Union[float, list], memory: Union[int, float], maxhyp: Union[int, float],
                 obsnz: Union[float, list],
                 max_ncomp: int, beta: float,
                 predscale: float, change_decision_threshold: float):
        """
        Complete representation of a single D-REX run.

        Parameters
        ----------
        input_sequence
            The input sequence, shape: (time, feature)
        prior
            The prior distribution to use
        hazard
            Hazard rate(s): scalar or list of numbers (for each input datum)
        memory
            Number of most-recent hypotheses to calculate
        maxhyp
            Number of hypotheses to calculate at every time step
        obsnz
            Observation noise, shape: (feature,)
        max_ncomp
            Maxmimum number of components (relevant if using a GMM prior)
        beta
            Threshold for new components (relevant if using a GMM prior
        predscale
            Predscale (D-REX internal)
        change_decision_threshold
            Threshold used for D-REX's change detector
        """
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

        self.input_sequence = transform_to_unified_drex_input_sequence_representation(input_sequence)
        self.prior = prior
        self.hazard = hazard
        self.obsnz = obsnz
        self.memory = memory
        self.maxhyp = maxhyp
        self.max_ncomp = max_ncomp
        self.beta = beta
        self.predscale = predscale
        self.change_decision_threshold = change_decision_threshold


class DREXResultsFilePsi:
    def __init__(self, predictions: dict, positions: dict):
        """
        Representation of D-REX's marginal predictive probability distribution (Psi)

        Parameters
        ----------
        predictions
            Predictions as dictionary, where each key corresponds to a feature, and its
            value corresponds to a numpy-array of shape (time,position)
        positions
            Prediction positions ass dictionary, where each key corresponds to a feature, and
            its value corresponds to a list with the numeric position values
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

    def features(self) -> list:
        """
        Return available feature indexes

        Returns
        -------
        list
            Feature indexes
        """
        return self._features

    def prediction_by_feature(self, feature: int) -> np.ndarray:
        """
        Return the predictive probability distribution of a single feature.

        Parameters
        ----------
        feature
            Feature index

        Returns
        -------
        np.ndarray
            Predictions, shape: (time, position)
        """
        return self._feature_to_predictions[feature]

    def positions_by_feature(self, feature: int) -> list:
        """
        Return the prediction positions of a single feature.

        Parameters
        ----------
        feature
            Feature index

        Returns
        -------
        np.ndarray
            Prediction positions
        """
        return self._feature_to_positions[feature]


class DREXResultsFile(ResultsFile):
    @staticmethod
    def save(results_file: DREXResultsFile, file_path: Union[str, Path]):
        raise NotImplementedError  # TODO

    @staticmethod
    def _load_prediction_results(prediction_results: dict) -> DREXResultsFilePsi:
        predictions = dict()
        positions = dict()

        nfeature = len(prediction_results)

        for f in range(nfeature):
            predictions[f] = np.array(
                prediction_results[f][0]["prediction"][0][0])  # convert to np.array with shape (time,position)

            f_positions = prediction_results[f][0]["positions"][0][0]
            if isinstance(f_positions, int) or isinstance(f_positions, float):  # convert
                positions[f] = np.array([f_positions])
            elif isinstance(f_positions, np.ndarray):
                positions[f] = np.array(f_positions)[0]

        return DREXResultsFilePsi(predictions, positions)

    @staticmethod
    def _load_processed_prior(estimate_suffstat_results) -> Prior:
        if "estimate_suffstat_results" not in estimate_suffstat_results:
            raise ValueError("Missing dictionary key 'estimate_suffstat_results'!")
        if "distribution" not in estimate_suffstat_results:
            raise ValueError("Missing dictionary key 'distribution'!")
        distribution = estimate_suffstat_results["distribution"]
        es_data = estimate_suffstat_results["estimate_suffstat_results"]

        if distribution == DistributionType.GAUSSIAN.value:
            data_means = es_data["mu"][0][0]
            data_covariance = es_data["ss"][0][0]
            data_n = es_data["n"][0][0]

            if data_means.shape[0] != data_covariance.shape[0] or data_covariance.shape[0] != data_n.shape[0]:
                raise ValueError(
                    "estimate_suffstat_results invalid! The number of features between mu, ss, and/or n "
                    "is not identical.")
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
                    "estimate_suffstat_results invalid! The number of features between mu, ss, and/or n "
                    "is not identical.")
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
                    "estimate_suffstat_results invalid! The number of features between mu, ss, and/or n "
                    "is not identical.")
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

    @staticmethod
    def load(file_path: Union[str, Path]) -> Union[DREXResultsFile, Prior]:
        data = from_mat(file_path, simplify_cells=False) # TODO adapt code for "simplify_cells=True"

        prior = DREXResultsFile._load_processed_prior(data)
        if "run_DREX_model_results" not in data:
            return prior

        instructions_file_path = data["instructions_file_path"][0]
        input_sequence = data["input_sequence"]
        run_results = data["run_DREX_model_results"]
        bd_results = data["post_DREX_beliefdynamics_results"]
        cd_results = data["post_DREX_changedecision_results"]
        pred_results = data["post_DREX_prediction_results"]

        input_sequence = transform_to_rundrexmodel_representation(
            transform_to_unified_drex_input_sequence_representation([input_sequence.T.tolist()]))
        surprisal = np.array(run_results["surprisal"][0][0])
        joint_surprisal = np.array(run_results["joint_surprisal"][0][0]).flatten()
        context_beliefs = np.array(run_results["context_beliefs"][0][0])
        belief_dynamics = np.array(bd_results).flatten()
        change_decision_changepoint = float(cd_results["changepoint"][0][0])
        change_decision_probability = np.array(cd_results["changeprobability"][0][0]).flatten()
        change_decision_threshold = float(data["change_decision_threshold"])
        if prior.distribution_type() in [DistributionType.GAUSSIAN, DistributionType.LOGNORMAL, DistributionType.GMM]:
            psi = DREXResultsFile._load_prediction_results(pred_results)
        else:
            psi = DREXResultsFilePsi({}, {})

        return DREXResultsFile(str(instructions_file_path), input_sequence, prior, surprisal,
                               joint_surprisal,
                               context_beliefs, belief_dynamics, change_decision_changepoint,
                               change_decision_probability,
                               change_decision_threshold, psi)

    # TODO add prediction_params from run_DREX_model.m?
    def __init__(self, instructions_file_path: Union[str, Path],
                 input_sequence: np.ndarray, prior: Prior,
                 surprisal: np.ndarray, joint_surprisal: np.ndarray, context_beliefs: np.ndarray,
                 belief_dynamics: np.ndarray, change_decision_changepoint: float,
                 change_decision_probability: np.ndarray, change_decision_threshold: float, psi: DREXResultsFilePsi):
        """
        Representation of a D-REX run.

        Parameters
        ----------
        instructions_file_path
            Where the instructions are stored
        input_sequence
            The processed input sequence, shape: (time, feature)
        prior
            The used prior distribution
        surprisal
            The computed surprisal values, shape: (time,feature)
        joint_surprisal
            The computed joint surprisal values, shape: (time,)
        context_beliefs
            The computed context belief values, shape: (memory, context)
        belief_dynamics
            The computed belief dynamics values, shape: (time,)
        change_decision_changepoint
            Change detector's calculated changepoint (although being an int, this value is stored as float
            to support float('nan'))
        change_decision_probability
            The computed change probability, shape: (time,)
        change_decision_threshold
            The threshold used by the change detector
        psi
            The marginal predictive probability distribution (Psi)
        """
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
                input_sequence_times == surprisal_times and surprisal_times == joint_surprisal_times and
                joint_surprisal_times == (belief_dynamics_times - 1)):
            raise ValueError(
                "Dimension 'time' invalid! Value must be equal for input_sequence({}), surprisal({}), "
                "joint_surprisal({}), and belief_dynamics({}-1).".format(
                    input_sequence_times, surprisal_times, joint_surprisal_times, belief_dynamics_times))
        if not (input_sequence_features == surprisal_features):
            raise ValueError(
                "Dimension 'feature' invalid! Value must be equal for input_sequence, and surprisal.")

        self.dimension_values = dict()
        self.dimension_values["time"] = input_sequence_times
        self.dimension_values["feature"] = input_sequence_features
        self.dimension_values["context"] = context_beliefs_contexts

        self.instructions_file_path = instructions_file_path
        self.prior = prior
        self.input_sequence = input_sequence
        self.surprisal = surprisal
        self.joint_surprisal = joint_surprisal
        self.context_beliefs = context_beliefs
        self.belief_dynamics = belief_dynamics
        self.change_decision_changepoint = change_decision_changepoint
        self.change_decision_probability = change_decision_probability
        self.change_decision_threshold = change_decision_threshold
        self.psi = psi
