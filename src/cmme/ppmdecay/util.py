import numbers


def list_to_str(lst, sep=", "):
    return sep.join(map(str, lst))


def str_to_list(s, sep=", "):
    return s.split(sep)


def auto_convert_input_sequence(input_sequence: list):
    """
    Converts an input sequence as nested list, as interpretable by the intermediate script.
    For instance:
    1) (shallow) listA => [listA]
    2) [listA, listB, ...] => id.
    :param input_sequence: A list
    :return:
    """
    if not isinstance(input_sequence, list) or len(input_sequence) == 0:
        raise ValueError("input_sequence invalid! Instance of list with at least one element expected.")
    first_element = input_sequence[0]
    if isinstance(first_element, list):  # assume already converted list
        return input_sequence
    elif isinstance(first_element, numbers.Number) or isinstance(first_element, str):  # assume shallow list
        return [input_sequence]
    else:
        return ValueError("input_sequence invalid! First element must be either a number, a character/string, "
                          "or a list.")
