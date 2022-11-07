from abc import ABC, abstractmethod
import numpy as np

from .distribution import Distribution
from .util import auto_convert_input_sequence


class Prior(ABC):
    """
    D-REX prior (at single time).
    """
    def __init__(self):
        pass

    @abstractmethod
    def distribution(self):
        pass

    @abstractmethod
    def features_count(self):
        pass

    @abstractmethod
    def D(self):
        pass


class GaussianPrior(Prior):
    def __init__(self, _mu, _ss, _n):
        super()

        # Check shapes
        if len(_mu.shape) != 2:
            raise ValueError("Shape of _mu invalid! Expected two dimensions: feature, D.")
        if len(_ss.shape) != 3:
            raise ValueError("Shape of _ss invalid! Expected three dimensions: feature, D, D.")
        if len(_n.shape) != 1:
            raise ValueError("Shape of _n invalid! Expected one dimension: feature.")
        [mu_features, mu_Ds] = _mu.shape
        [ss_features, ss_firstDs, ss_secondDs] = _ss.shape
        [n_features] = _n.shape

        # Check for dimension equality
        if not (mu_features == ss_features and ss_features == n_features):
            raise ValueError("Dimension 'feature' invalid! Value must be equal for _mu, _ss, and _n.")
        if not (mu_Ds == ss_firstDs and ss_firstDs == ss_secondDs):
            raise ValueError("Dimension 'D' invalid! Value must be equal for _mu, and _ss.")

        # Set fields
        self.dimension_values = dict()
        """Dimensions present in this prior"""
        self.dimension_values["feature"] = mu_features
        self.dimension_values["D"] = mu_Ds

        self.mu = np.array(_mu, dtype=float)
        """Mean: (feature, D) => 1"""

        self.ss = np.array(_ss, dtype=float)
        """Covariance: feature => DxD"""

        self.n = np.array(_n, dtype=int)
        """Number of observations summarized in the parameters: feature => 1"""

    def distribution(self):
        return Distribution.GAUSSIAN

    def features_count(self):
        return self.dimension_values["feature"]

    def D(self):
        return self.dimension_values["D"]

class LognormalPrior(GaussianPrior):
    """Parameters equivalent to GaussianPrior"""
    def __init__(self):
        super()

    def distribution(self):
        return Distribution.LOGNORMAL


class GmmPrior(Prior):
    """D-REX sets D=1."""
    def __init__(self, _mu, _sigma, _n, _pi, _sp, _k):
        super()
        # Check shapes
        if len(_mu.shape) != 2:
            raise ValueError("Shape of _mu invalid! Expected two dimensions: feature, component.")
        if len(_sigma.shape) !=2:
            raise ValueError("Shape of _ss invalid! Expected three dimensions: feature, component")
        if len(_n.shape) != 2:
            raise ValueError("Shape of _n invalid! Expected one dimension: feature, component.")
        if len(_pi.shape) != 2:
            raise ValueError("Shape of _pi invalid! Expected one dimension: feature, component.")
        if len(_sp.shape) != 2:
            raise ValueError("Shape of _sp invalid! Expected one dimension: feature, component.")
        if len(_k.shape) != 2:
            raise ValueError("Shape of _k invalid! Expected one dimension: feature, component.")

        # Check for dimension equality
        [mu_features, mu_components] = _mu.shape
        [sigma_features, sigma_components] = _sigma.shape
        [n_features, n_components] = _n.shape
        [pi_features, pi_components] = _pi.shape
        [sp_features, sp_components] = _sp.shape
        [k_features] = _k.shape

        if not (mu_features == sigma_features and sigma_features == n_features and n_features == pi_features and
                pi_features == sp_features and sp_features == k_features):
            raise ValueError("Dimension 'feature' invalid! Value must be equal for _mu, _sigma, _n, _pi, _sp, and_k.")
        if not (mu_components == sigma_components and sigma_components == n_components and n_components == pi_components and
                pi_components == sp_components):
            raise ValueError("Dimension 'component' invalid! Value must be equal for _mu, _sigma, _n, _pi, and _sp.")

        self.dimension_values = dict()
        self.dimension_values["feature"] = mu_features
        self.dimension_values["component"] = mu_components
        self.dimension_values["D"] = 1 # TODO Is it appropriate to use a fixed value?

        self.mu = np.array(_mu, dtype=float)
        """Component's Gaussian parameter mean: feature, component => 1 """
        self.sigma = np.array(_sigma, dtype=float)
        """Component's Gaussian parameter covariance matrix: feature, component => 1x1 """
        self.n = np.array(_n, dtype=int)
        """Component's Gaussian parameter n: feature, component => 1"""
        self.pi = np.array(_pi, dtype=float)
        """Component's weight: feature, component => 1 """
        self.sp = np.array(_sp, dtype=float)
        """Component's likelihood: feature, component => 1"""
        self.k = np.array(_k, dtype=int)
        """Number of components: feature => 1"""

    def distribution(self):
        return Distribution.GMM

    def features_count(self):
        return self.dimension_values["feature"]

    def D(self):
        return self.dimension_values["D"]

    def max_n_comp(self):
        return self.dimension_values["component"]

class PoissonPrior(Prior):
    def __init__(self, _lambda, _n, _D):
        super()
        # Check shapes
        if len(_lambda) != 1:
            raise ValueError("Shape of _lambda invalid! Expected one dimension: feature.")
        if len(_n) != 1:
            raise ValueError("Shape of _n invalid! Expected one dimension: feature.")

        # Check for dimension equality
        [lambda_features] = _lambda.shape
        [n_features] = _n.shape

        if not lambda_features == n_features:
            raise ValueError("Dimension 'feature' invalid! Value must be equal for _lambda and _n.")

        self.dimension_values = dict()
        self.dimension_values["feature"] = lambda_features

        self.lambd = _lambda
        """Lambda parameter: feature => 1"""
        self.n = _n
        """Observation count: feature => 1"""
        self.D = _D
        """Interval size: feature => 1"""

    def distribution(self):
        return Distribution.POISSON

    def features_count(self):
        return self.dimension_values["feature"]

    def D(self):
        return self._D

class UnprocessedPrior(Prior):
    def __init__(self, distribution: Distribution, D, prior_input_sequence, max_n_comp = None):
        """
        Creates an unprocessed prior which will be processed by D-REX.
        :param distribution: Distribution
        :param D: positive int
        :param prior_input_sequence: np.array with shape (time, feature), or 2d-array with feature x time
        :param max_n_comp: positive int
        """
        pis = auto_convert_input_sequence(prior_input_sequence)

        # Check prior_input_sequence
        if len(pis.shape) != 2:
            raise ValueError("Shape of prior_input_sequence invalid! Expected two dimensions: time, feature")
        [prior_input_sequence_times, prior_input_sequence_features] = pis.shape

        # Check D
        if distribution == Distribution.GMM:
            if D != 1:
                raise ValueError("D invalid! For distribution=GMM, D must be equal to 1.")
        if D < 1:
            raise ValueError("D invalid! Value must be greater than or equal 1.")
        if prior_input_sequence_times < D:
            raise ValueError("D invalid! Value must be less than the number of observations in prior_input_sequence_times.")

        # Check max_n_comp
        if distribution == Distribution.GMM:
            if max_n_comp < 1:
                raise ValueError("max_n_comp invalid! Value must be greater than or equal 1.")

        self.dimension_values = dict()
        self.dimension_values["D"] = D
        self.dimension_values["feature"] = prior_input_sequence_features

        # Set attributes
        self._distribution = distribution
        self._prior_input_sequence = pis
        self._max_n_comp = max_n_comp

    def distribution(self):
        return self._distribution

    def D(self):
        return self.dimension_values["D"]

    def features_count(self):
        return self.dimension_values["feature"]


class PriorMatrix:
    def __init__(self):
        pass