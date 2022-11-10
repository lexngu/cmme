from datetime import datetime

from cmme.config import Config


def list_to_str(l, sep=", "):
    return sep.join(map(str, l))


def str_to_list(s, sep=", "):
    return s.split(sep)

def ppmdecay_default_instructions_file_path(alias = None):
    instructions_file_filename = datetime.now().isoformat().replace("-", "").replace(":", "").replace(".", "")
    instructions_file_filename = instructions_file_filename + "-ppmdecay-instructionsfile"
    instructions_file_filename = instructions_file_filename + "-" + alias if alias is not None else instructions_file_filename
    instructions_file_filename = instructions_file_filename + ".csv"
    return Config().model_io_path() / instructions_file_filename

def ppmdecay_default_results_file_path(alias = None):
    results_file_filename = datetime.now().isoformat().replace("-", "").replace(":", "").replace(".", "")
    results_file_filename = results_file_filename + "-ppmdecay-resultsfile"
    results_file_filename = results_file_filename + "-" + alias if alias is not None else results_file_filename
    results_file_filename = results_file_filename + ".csv"
    return Config().model_io_path() / results_file_filename