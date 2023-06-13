from pathlib import Path
from typing import List
import os


def flatten_list(obj: List, recursive=False) -> List:
    """
    Returns a list with all elements of +obj+. If an element is a list, this element gets unpacked.
    :param obj:
    :param recursive:
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
                result = [*result, *o]
        else:
            result.append(o)
    return result


def path_as_string_with_trailing_slash(path) -> str:
    """
    Takes a Path or any object that can be converted into a Path object and returns a Path for which a trailing slash is ensured.
    :param path:
    :return:
    """
    if path is None:
        raise ValueError("path must not be None!")
    if not isinstance(path, Path):
        path = Path(path)
    try:
        path = path.expanduser().resolve()
    except:
        pass
    return os.path.join(path, '')
