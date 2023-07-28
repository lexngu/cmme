import numbers
from pathlib import Path

from cmme.config import Config


def list_to_str(lst, sep=", "):
    return sep.join(map(str, lst))


def str_to_list(s, sep=", "):
    return s.split(sep)


def ppmdecay_default_instructions_file_path(alias: str = None, io_path: Path = Config().model_io_path()) -> str:
    instructions_file_filename = "ppmdecay-instructionsfile"
    instructions_file_filename = instructions_file_filename + "-" + alias if alias is not None \
        else instructions_file_filename
    instructions_file_filename = instructions_file_filename + ".feather"
    return str(io_path / instructions_file_filename)


def ppmdecay_default_results_file_path(alias: str = None, io_path: Path = Path(".")) -> str:
    results_file_filename = "ppmdecay-resultsfile"
    results_file_filename = results_file_filename + "-" + alias if alias is not None else results_file_filename
    results_file_filename = results_file_filename + ".feather"
    return str(io_path / results_file_filename)


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
