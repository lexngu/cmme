import pytest

from cmme.idyom.base import BasicViewpoint, transform_viewpoints_list_to_string_list, DerivedViewpoint, ThreadedViewpoint


def test_viewpoints_list_to_string_list():
    lst = None
    with pytest.raises(ValueError):
        transform_viewpoints_list_to_string_list(lst)

    lst = []
    with pytest.raises(ValueError):
        transform_viewpoints_list_to_string_list(lst)

    lst = [BasicViewpoint.CPITCH]
    result = transform_viewpoints_list_to_string_list(lst)
    assert result == ['cpitch']

    lst = [BasicViewpoint.CPITCH, [BasicViewpoint.CPITCH, BasicViewpoint.ONSET]]
    result = transform_viewpoints_list_to_string_list(lst)
    assert result == ['cpitch', ['cpitch', 'onset']]

    lst = [BasicViewpoint.CPITCH, [[DerivedViewpoint.IOI, ThreadedViewpoint.THR_CPINT_FIB], BasicViewpoint.ONSET]]
    result = transform_viewpoints_list_to_string_list(lst)
    assert result == ['cpitch', [['ioi', 'thr-cpint-fib'], 'onset']]
