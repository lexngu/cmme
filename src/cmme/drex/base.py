from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from .util import transform_to_unified_drex_input_sequence_representation


class DistributionType(Enum):
    """Distribution types implemented by D-REX"""
    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    GMM = "gmm"
    POISSON = "poisson"


class Prior(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def distribution_type(self):
        raise NotImplementedError

    @abstractmethod
    def feature_count(self):
        raise NotImplementedError

    @abstractmethod
    def D_value(self):
        raise NotImplementedError


class GaussianPrior(Prior):
    def __init__(self, means: np.ndarray, covariance: np.ndarray, n: np.ndarray):
        """
        Representation of a Gaussian prior

        Parameters
        ----------
        means
            mean values, shape: (feature, D)
        covariance
            covariance values, shape: (feature, D, D)
        n
            indicator of how many elements were counted so far, shape: (feature,)
        """

        # Ensure parameters to have consistent shapes
        super().__init__()
        if len(means.shape) != 2:
            raise ValueError("Shape of means invalid! Expected two dimensions: feature, D.")
        if len(covariance.shape) != 3:
            raise ValueError("Shape of covariance invalid! Expected three dimensions: feature, D, D.")
        if len(n.shape) != 1:
            raise ValueError("Shape of n invalid! Expected one dimension: feature.")
        [means_feature_count, means_D_value] = means.shape
        [covariance_feature_count, covariance_first_D_value, covariance_second_D_value] = covariance.shape
        [n_feature_count] = n.shape
        if not (means_feature_count == covariance_feature_count and covariance_feature_count == n_feature_count):
            raise ValueError("Dimension 'feature' invalid! Value must be equal for parameters "
                             "means, covariance, and n.")
        if not (means_D_value == covariance_first_D_value and covariance_first_D_value == covariance_second_D_value):
            raise ValueError("Dimension 'D' invalid! Value must be equal for parameters means, and covariance.")

        self._feature_count = means_feature_count
        """Number of features"""

        self._D_value = means_D_value
        """D value (amount of temporal dependence while calculating the conditional distribution)"""

        self.means = means
        self.covariance = covariance
        self.n = n

    def distribution_type(self):
        return DistributionType.GAUSSIAN

    def feature_count(self):
        return self._feature_count

    def D_value(self):
        return self._D_value


class LognormalPrior(Prior):
    def __init__(self, means: np.ndarray, covariance: np.ndarray, n: np.ndarray):
        """
        Representation of a Lognormal prior

        Parameters
        ----------
        means
            mean values, shape: (feature, D)
        covariance
            covariance values, shape: (feature, D, D)
        n
            indicator of how many elements were counted so far, shape: (feature,)
        """

        # Ensure parameters to have consistent shapes
        super().__init__()
        if len(means.shape) != 2:
            raise ValueError("Shape of means invalid! Expected two dimensions: feature, D.")
        if len(covariance.shape) != 3:
            raise ValueError("Shape of covariance invalid! Expected three dimensions: feature, D, D.")
        if len(n.shape) != 1:
            raise ValueError("Shape of n invalid! Expected one dimension: feature.")
        [means_feature_count, means_D_value] = means.shape
        [covariance_feature_count, covariance_first_D_value, covariance_second_D_value] = covariance.shape
        [n_feature_count] = n.shape
        if not (means_feature_count == covariance_feature_count and covariance_feature_count == n_feature_count):
            raise ValueError("Dimension 'feature' invalid! Value must be equal for parameters "
                             "means, covariance, and n.")
        if not (means_D_value == covariance_first_D_value and covariance_first_D_value == covariance_second_D_value):
            raise ValueError("Dimension 'D' invalid! Value must be equal for parameters means, and covariance.")

        self._feature_count = means_feature_count
        """Number of features"""

        self._D_value = means_D_value
        """D value (amount of temporal dependence while calculating the conditional distribution)"""

        self.means = means
        self.covariance = covariance
        self.n = n

    def distribution_type(self):
        return DistributionType.LOGNORMAL

    def feature_count(self):
        return self._feature_count

    def D_value(self):
        return self._D_value


class GmmPrior(Prior):
    def __init__(self, means: np.ndarray, covariance: np.ndarray, n: np.ndarray,
                 pi: np.ndarray, sp: np.ndarray, k: np.ndarray):
        """
        Representation of a Gaussian Mixture Model prior

        Parameters
        ----------
        means
            mean values, shape: (feature, component)
        covariance
            covariance values, shape: (feature, component)
        n
            indicator of how many elements were counted so far, shape: (feature, component)
        pi
        sp
        k
        """

        # Check shapes
        super().__init__()
        if len(means.shape) != 2:
            raise ValueError("Shape of means invalid! Expected two dimensions: feature, component.")
        if len(covariance.shape) != 2:
            raise ValueError("Shape of ss invalid! Expected two dimensions: feature, component")
        if len(n.shape) != 2:
            raise ValueError("Shape of n invalid! Expected two dimensions: feature, component.")
        if len(pi.shape) != 2:
            raise ValueError("Shape of pi invalid! Expected two dimensions: feature, component.")
        if len(sp.shape) != 2:
            raise ValueError("Shape of sp invalid! Expected two dimensions: feature, component.")
        if len(k.shape) != 1:
            raise ValueError("Shape of k invalid! Expected one dimension: feature.")

        # Check for dimension equality
        [means_feature_count, means_component_count] = means.shape
        [covariance_feature_count, covariance_component_count] = covariance.shape
        [n_feature_count, n_component_count] = n.shape
        [pi_feature_count, pi_component_count] = pi.shape
        [sp_feature_count, sp_component_count] = sp.shape
        [k_feature_count] = k.shape

        if not (means_feature_count == covariance_feature_count and covariance_feature_count == n_feature_count and
                n_feature_count == pi_feature_count and pi_feature_count == sp_feature_count and
                sp_feature_count == k_feature_count):
            raise ValueError("Dimension feature invalid! Value must be equal for means, covariance, n, pi, sp, and k.")
        if not (means_component_count == covariance_component_count and
                covariance_component_count == n_component_count and
                n_component_count == pi_component_count and pi_component_count == sp_component_count):
            raise ValueError("Dimension component invalid! Value must be equal for means, covariance, n, pi, and sp.")

        self._feature_count = means_feature_count
        """Number of features"""

        self.means = means
        self.covariance = covariance
        self.n = n
        self.pi = pi
        self.sp = sp
        self.k = k

    def distribution_type(self):
        return DistributionType.GMM

    def feature_count(self):
        return self._feature_count

    def D_value(self):
        return 1


class PoissonPrior(Prior):
    def __init__(self, lambd: np.ndarray, n: np.ndarray):
        """
        Representation of a Poisson prior

        Parameters
        ----------
        lambd
            interval size, shape: (feature,)
        n
            indicator of how many elements were counted so far, shape: (feature,)
        """
        super().__init__()
        if len(lambd.shape) != 1:
            raise ValueError("Shape of lambd invalid! Expected one dimension: feature.")
        if len(n.shape) != 1:
            raise ValueError("Shape of n invalid! Expected one dimension: feature.")

        # Check for dimension equality
        [lambda_features] = lambd.shape
        [n_features] = n.shape
        if not lambda_features == n_features:
            raise ValueError("Dimension 'feature' invalid! Value must be equal for lambd and n.")

        self.lambd = lambd
        self.n = n
        self._feature_count = lambda_features

    def distribution_type(self):
        return DistributionType.POISSON

    def feature_count(self):
        return self._feature_count

    def D_value(self):
        return NotImplementedError


class UnprocessedPrior(Prior):
    def __init__(self, distribution: DistributionType, prior_input_sequence: np.ndarray,
                 D: int = None):
        """
        Representation of an unprocessed prior which will be processed by D-REX and used as "prior" for
        new context window hypotheses.

        Parameters
        ----------
        distribution
            Distribution type
        prior_input_sequence
            np.array with shape (time, feature), or 2d-list with feature x time
        D
            Amount of temporal dependence. If None, D-REX's default value will be used (Gaussian: 1, Poisson: 50),
            if *distribution* is GMM, D=1 is enforced.
        """

        super().__init__()
        pis = transform_to_unified_drex_input_sequence_representation(prior_input_sequence)

        # Check prior_input_sequence
        if len(pis.shape) == 3:
            [prior_input_sequence_trials, prior_input_sequence_times, prior_input_sequence_features] = pis.shape
        elif len(pis.shape) == 1:
            [prior_input_sequence_trials] = pis.shape
            pis_first_element = pis[0]
            if len(pis_first_element.shape) == 1:
                [prior_input_sequence_times] = pis[0].shape
                prior_input_sequence_features = 1
            elif len(pis_first_element.shape) == 2:
                [prior_input_sequence_times, prior_input_sequence_features] = pis[0].shape
            else:
                raise ValueError("Prior input sequence has an invalid shape!")
        else:
            raise ValueError("Could not convert input sequence.")

        # Check D
        if distribution == DistributionType.GMM:
            if D is None:
                D = 1
            if D != 1:
                raise ValueError("D invalid! For distribution=GMM, D must be equal to 1.")
        elif distribution == DistributionType.GAUSSIAN:
            if D is None:
                D = 1
        elif distribution == DistributionType.POISSON:
            if D is None:
                D = 50
        if D is None or D < 1:
            raise ValueError("D invalid! Value must be greater than or equal 1.")
        if prior_input_sequence_times < D:  # TODO check against min times
            raise ValueError("D invalid! Value must be less than the number of observations in prior_input_sequence.")

        self._D = D
        self._feature_count = prior_input_sequence_features
        self._trials_count = prior_input_sequence_trials

        # Set attributes
        self._distribution = distribution
        self.prior_input_sequence = pis

    def distribution_type(self):
        return self._distribution

    def D_value(self):
        return self._D

    def feature_count(self):
        return self._feature_count

    def trials_count(self):
        return self._trials_count
