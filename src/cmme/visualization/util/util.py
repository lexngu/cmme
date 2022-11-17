from datetime import datetime

import numpy as np

from cmme.config import Config

def cmme_default_plot_instructions_file_path(alias = None):
    plot_file_filename = datetime.now().isoformat().replace("-", "").replace(":", "").replace(".", "")
    plot_file_filename = plot_file_filename + "-plot-input"
    plot_file_filename = plot_file_filename + "-" + alias if alias is not None else plot_file_filename
    plot_file_filename = plot_file_filename + ".mat"
    return Config().model_io_path() / plot_file_filename
