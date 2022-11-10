import csv
import os
from pathlib import Path

# R_HOME specifies the R instance to use by rpy2.
# Needs to happen before any imports from rpy2
from cmme.config import Config

os.environ["R_HOME"] = str(Config().r_home())
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
PPM_RUN_FILEPATH = (Path(__file__).parent.parent.parent.parent.parent.absolute() / "./res/wrappers/ppm-decay/ppmdecay_intermediate_script.R").resolve()


def invoke_model(instructions_file_path : Path):
    """

    :param instructions_file_path:
    :return: R console output
    """
    with open(PPM_RUN_FILEPATH) as f:
        r_file_contents = f.read()
    package = SignatureTranslatedAnonymousPackage(r_file_contents, "ppm-python-bridge")

    results_file_path = str(package.ppmdecay_intermediate_script(str(instructions_file_path)))

    return results_file_path
