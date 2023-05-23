import pytest

from cmme.idyom.base import BasicViewpoint, viewpoints_list_to_string_list, DerivedViewpoint, ThreadedViewpoint


def test_viewpoints_list_to_string_list():
    lst = None
    with pytest.raises(ValueError):
        viewpoints_list_to_string_list(lst)

    lst = []
    with pytest.raises(ValueError):
        viewpoints_list_to_string_list(lst)

    lst = [BasicViewpoint.CPITCH]
    result = viewpoints_list_to_string_list(lst)
    assert result == ['cpitch']

    lst = [BasicViewpoint.CPITCH, [BasicViewpoint.CPITCH, BasicViewpoint.ONSET]]
    result = viewpoints_list_to_string_list(lst)
    assert result == ['cpitch', ['cpitch', 'onset']]

    lst = [BasicViewpoint.CPITCH, [[DerivedViewpoint.IOI, ThreadedViewpoint.THR_CPINT_FIB], BasicViewpoint.ONSET]]
    result = viewpoints_list_to_string_list(lst)
    assert result == ['cpitch', [['ioi', 'thr-cpint-fib'], 'onset']]
