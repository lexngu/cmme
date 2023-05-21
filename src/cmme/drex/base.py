from abc import ABC, abstractmethod
from enum import Enum
import numpy.typing as npt

from .util import auto_convert_input_sequence


class DistributionType(Enum):
    """Implemented distribution types of D-REX"""
    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    GMM = "gmm"
    POISSON = "poisson"


class Prior(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def distribution_type(self):
        pass

    @abstractmethod
    def feature_count(self):
        pass

    @abstractmethod
    def D_value(self):
        pass


class GaussianPrior(Prior):
    def __init__(self, means: npt.ArrayLike, covariance: npt.ArrayLike, n: npt.ArrayLike):
        # Ensure parameters to have consistent shapes
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
            raise ValueError("Dimension 'feature' invalid! Value must be equal for parameters means, covariance, and n.")
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
    def __init__(self, means: npt.ArrayLike , covariance: npt.ArrayLike, n: npt.ArrayLike):
        # Ensure parameters to have consistent shapes
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
            raise ValueError("Dimension 'feature' invalid! Value must be equal for parameters means, covariance, and n.")
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
        DistributionType.LOGNORMAL

    def feature_count(self):
        return self._feature_count

    def D_value(self):
        return self._D_value


class GmmPrior(Prior):
    """
    1-variate Gaussian Mixture Model, consisting of up to +k+ components.
    """
    def __init__(self, means, covariance, n, pi, sp, k):
        # Check shapes
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
                n_feature_count == pi_feature_count and pi_feature_count == sp_feature_count and sp_feature_count == k_feature_count):
            raise ValueError("Dimension feature invalid! Value must be equal for means, covariance, n, pi, sp, and k.")
        if not (means_component_count == covariance_component_count and covariance_component_count == n_component_count and
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
    def __init__(self, lambd, n):
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
        pass # TODO


class UnprocessedPrior(Prior):
    def __init__(self, distribution: DistributionType, prior_input_sequence, D=None, max_n_comp=None, beta=None): # TODO move beta to D-REX hyper parameters
        """
        Creates an unprocessed prior which will be processed by D-REX and used as "prior" for new context window hypotheses.
        :param distribution: DistributionType
        :param prior_input_sequence: np.array with shape (time, feature), or 2d-array with feature x time
        :param D: amount of temporal dependence. If None, D-REX's default value will be used (Gaussian: 1, Poisson: 50), if *distribution* is GMM, D=1 is enforced.
        :param max_n_comp: Relevant for *distribution* GMM: Maxmimum number of components in Gaussian Mixture Model (default: 10)
        :param beta: probability between [0,1]. Threshold for new GMM components (see D-REX).
        """

        pis = auto_convert_input_sequence(prior_input_sequence)

        # Check prior_input_sequence
        [prior_input_sequence_trials, prior_input_sequence_times, prior_input_sequence_features] = pis.shape

        # Check D
        if distribution == DistributionType.GMM:
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
        if prior_input_sequence_times < D: # TODO check against min times
            raise ValueError("D invalid! Value must be less than the number of observations in prior_input_sequence.")

        # Check max_n_comp
        if distribution == DistributionType.GMM:
            if max_n_comp is None:
                max_n_comp = 10
            if max_n_comp < 1:
                raise ValueError("max_n_comp invalid! Value must be greater than or equal 1.")

        self._D = D
        self._feature_count = prior_input_sequence_features
        self._trials_count = prior_input_sequence_trials

        # Set attributes
        self._distribution = distribution
        self._prior_input_sequence = pis
        self._max_n_comp = max_n_comp
        self._beta = beta

    def distribution_type(self):
        return self._distribution

    def D_value(self):
        return self._D

    def feature_count(self):
        return self._feature_count

    def trials_count(self):
        return self._trials_count
