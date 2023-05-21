import pytest
from numpy import array, nan, ndarray

from cmme.drex.base import UnprocessedPrior, DistributionType, GaussianPrior, GmmPrior


def test_unprocessed_prior_with_prior_input_sequence_none_fails():
    with pytest.raises(ValueError):
        UnprocessedPrior(DistributionType.GAUSSIAN, None)

    with pytest.raises(ValueError):
        UnprocessedPrior(DistributionType.GAUSSIAN, [])


def test_unprocessed_prior_prior_input_sequence_should_be_nparray():
    pis = [1, 2, 3]
    p = UnprocessedPrior(DistributionType.GAUSSIAN, pis)
    assert isinstance(p._prior_input_sequence, ndarray)
    assert p._prior_input_sequence.shape == (1, 3, 1)
    assert p.feature_count() == 1
    assert p.trials_count() == 1

    pis = [[[1, 2, 3], [11, 12, 13], [21, 22, 23]]]
    p = UnprocessedPrior(DistributionType.GAUSSIAN, pis)
    assert isinstance(p._prior_input_sequence, ndarray)
    assert p._prior_input_sequence.shape == (1, 3, 3)
    assert p.feature_count() == 3
    assert p.trials_count() == 1

    pis = array([
        [[1, 2, 3], [11, 12, 13], [21, 22, 23]],
        [[31, 32, 33], [41, 42, 43], [51, 52, 53]]
    ], dtype=object)
    p = UnprocessedPrior(DistributionType.GAUSSIAN, pis)
    assert isinstance(p._prior_input_sequence, ndarray)
    assert p._prior_input_sequence.shape == (2, 3, 3)
    assert p.feature_count() == 3
    assert p.trials_count() == 2

def test_gaussian_prior():
    means = array([[72.09996676, 72.09996676]])
    covariance = array([[[60.32023955, 47.40748402],
        [47.40748402, 60.32023955]]])
    n = array([2])

    p = GaussianPrior(means, covariance, n)
    assert p.D_value() == 2

def test_gmm_prior():
    means = array([[ 2., nan, nan, nan, nan, nan, nan, nan, nan, nan],
       [ 2., nan, nan, nan, nan, nan, nan, nan, nan, nan],
       [ 2., nan, nan, nan, nan, nan, nan, nan, nan, nan]])
    covariance = array([[0.66666667,        nan,        nan,        nan,        nan,
               nan,        nan,        nan,        nan,        nan],
       [0.66666667,        nan,        nan,        nan,        nan,
               nan,        nan,        nan,        nan,        nan],
       [0.66666667,        nan,        nan,        nan,        nan,
               nan,        nan,        nan,        nan,        nan]])
    n = array([[ 1., nan, nan, nan, nan, nan, nan, nan, nan, nan],
       [ 1., nan, nan, nan, nan, nan, nan, nan, nan, nan],
       [ 1., nan, nan, nan, nan, nan, nan, nan, nan, nan]])
    pi = array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    sp = array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    k = array([1, 1, 1])

    p = GmmPrior(means, covariance, n, pi, sp, k)
    assert p.D_value() == 1

def test_poisson_prior():
    pass