import numbers


def list_to_str(lst, sep=", "):
    return sep.join(map(str, lst))


def str_to_list(s, sep=", "):
    return s.split(sep)


def auto_convert_input_sequence(seq: list):
    """
    Converts an input sequence as nested list, as interpretable by the intermediate script.
    For instance:
    1) (shallow) listA => [listA]
    2) [listA, listB, ...] => id.
    :param seq: A list
    :return:
    """
    if not isinstance(seq, list) or len(seq) == 0:
        raise ValueError("seq invalid! Instance of list with at least one element expected.")
    first_element = seq[0]
    if isinstance(first_element, list):  # assume already converted list
        return seq
    elif isinstance(first_element, numbers.Number) or isinstance(first_element, str):  # assume shallow list
        return [seq]
    else:
        return ValueError("seq invalid! First element must be either a number, a character/string, "
                          "or a list.")
