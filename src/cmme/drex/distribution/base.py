from enum import Enum


class DistributionType(Enum):
    """Implemented distribution types of D-REX"""
    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    GMM = "gmm"
    POISSON = "poisson"
