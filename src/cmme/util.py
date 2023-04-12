from res.packages.cl4py.cl4py import List


def flatten_list(obj: List(), recursive = False) -> List:
    """
    Returns a list with all elements of +obj+. If an element is a list, this element gets unpacked though.
    :param obj:
    :return:
    """
    if obj is None or obj == []:
        return []
    if not isinstance(obj, list):
        return [obj]

    result = []
    for o in obj:
        if isinstance(o, list):
            if recursive:
                result = [*result, *flatten_list(o)]
            else:
                result = [*result, o]
        else:
            result.append(o)
    return result