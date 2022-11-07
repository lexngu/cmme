from enum import Enum


class Distribution(Enum):
    """Distribution types supported by D-REX"""
    GAUSSIAN = "gaussian"
    LOGNORMAL = "lognormal"
    GMM = "gmm"
    POISSON = "poisson"
