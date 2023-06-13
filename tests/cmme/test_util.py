import tempfile
from pathlib import Path

import pytest

from cmme.util import flatten_list, path_as_string_with_trailing_slash


def test_flatten_list_always_returns_a_list():
    assert isinstance(flatten_list([]), list)
    assert isinstance(flatten_list(123), list)
    assert isinstance(flatten_list(None), list)
    assert isinstance(flatten_list([[]]), list)


def test_flatten_list_recursive():
    assert flatten_list([1, [2]], False) == [1, 2]
    assert flatten_list([1, [2]], True) == [1, 2]

    assert flatten_list([1, [2, [3, 4]]], False) == [1, 2, [3, 4]]
    assert flatten_list([1, [2, [3, 4]]], True) == [1, 2, 3, 4]


def test_path_as_string_with_trailing_slash():
    with pytest.raises(ValueError):
        path_as_string_with_trailing_slash(None)
    assert isinstance(path_as_string_with_trailing_slash(''), str)
    assert path_as_string_with_trailing_slash('/Volumes/Data/midi') == '/Volumes/Data/midi/'
    with tempfile.TemporaryDirectory() as tmpdir:
        assert path_as_string_with_trailing_slash(Path(tmpdir)) == (str(Path(tmpdir).expanduser().resolve()) + '/')