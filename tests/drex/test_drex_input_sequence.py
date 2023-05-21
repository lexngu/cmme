import pytest

from cmme.drex.util import auto_convert_input_sequence


def test_auto_convert_input_sequence():
    input_sequence = None
    with pytest.raises(ValueError):
        auto_convert_input_sequence(input_sequence)

    input_sequence = []
    with pytest.raises(ValueError):
        auto_convert_input_sequence(input_sequence)

    input_sequence = [1]
    result = auto_convert_input_sequence(input_sequence)
    assert result.shape == (1, 1, 1)

    input_sequence = [1, 2, 3]
    result = auto_convert_input_sequence(input_sequence)
    assert result.shape == (1, 3, 1)

    input_sequence = [[[1, 2, 3]], [[11, 12, 13]]] # 2 trials, 3 times, 1 feature
    result = auto_convert_input_sequence(input_sequence)
    assert result.shape == (2, 3, 1)

    input_sequence = [[[1, 2, 3], [11, 12, 13]]] # 1 trial, 3 times, 2 features
    result = auto_convert_input_sequence(input_sequence)
    assert result.shape == (1, 3, 2)

    input_sequence = [ # 2 trials, 3 times, 2 features
        [[1, 2, 3], [11, 12, 13]], # trial 1
        [[11, 12, 13], [111, 112, 113]] # trial 2
    ]
    result = auto_convert_input_sequence(input_sequence)
    assert result.shape == (2, 3, 2)