from datetime import datetime

import numpy as np

from cmme.config import Config

def cmme_default_plot_instructions_file_path(alias = None):
    plot_instructions_file_path = datetime.now().isoformat().replace("-", "").replace(":", "").replace(".", "")
    plot_instructions_file_path = plot_instructions_file_path + "-plot-input"
    plot_instructions_file_path = plot_instructions_file_path + "-" + alias if alias is not None else plot_instructions_file_path
    plot_instructions_file_path = plot_instructions_file_path + ".mat"
    return Config().model_io_path() / plot_instructions_file_path

def cmme_default_plot_output_file_path(alias = None):
    plot_output_file_path = datetime.now().isoformat().replace("-", "").replace(":", "").replace(".", "")
    plot_output_file_path = plot_output_file_path + "-plot-output"
    plot_output_file_path = plot_output_file_path + "-" + alias if alias is not None else plot_output_file_path
    return Config().model_io_path() / plot_output_file_path