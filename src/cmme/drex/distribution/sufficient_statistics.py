from abc import ABC

import numpy as np


class ParameterizedDistribution(ABC):
    """
    Parameterized distribution (in D-REX: prior, suffstat)
    """
    def __init__(self):
        pass


class MultivariateGaussianDistribution(ParameterizedDistribution):
    """
    D-variate Gaussian distribution, with +D+ mean values, and a DxD covariance matrix.
    If D = 1, one gets a univariate Gaussian distribution with one mean and one variance value.
    """

    def __init__(self, means, covariance):
        super()

        self.means = means
        """Mean of each variate"""

        self.covariance = covariance
        """Covariance matrix"""


class UpdateableMultivariateGaussianDistribution(MultivariateGaussianDistribution):
    def __init__(self, means, covariance, n):
        super(means, covariance)

        self.n = n
        """Number of observations summarized in the parameters so far"""


class UpdateableGmmDistribution(ParameterizedDistribution):
    def __init__(self, means, covariance, n, pi, sp, k):
        super()

        self.k = k
        """Number of components"""

        self.components = dict()
        """Components (Gaussian distributions)"""

        for component_index in range(k):
            component_mean = means[component_index]
            component_covariance = covariance[component_index]
            component_n = n[component_index]

            self.components = UpdateableMultivariateGaussianDistribution(component_mean, component_covariance, component_n)

        self.pi = np.array(pi, dtype=float)
        """Components' weight"""

        self.sp = np.array(sp, dtype=float)
        """Components' likelihood"""


class PoissonDistribution(ParameterizedDistribution):
    def __init__(self, lambd):
        super()

        self.lambd = lambd
        """Lambda parameter"""


class UpdateablePoissonDistribution(PoissonDistribution):
    def __init__(self, lambd, n, D):
        super(lambd)

        self.n = n
        """Observation count"""

        self.D = D
        """Interval size"""
