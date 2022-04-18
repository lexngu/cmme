import configparser
from pathlib import Path


class Config:
    DEFAULT_CONFIG_FILE_PATH = (Path(__file__).parent / "cmme-comparison.ini").resolve()

    def __init__(self, config_file_path: Path = None):
        self.configParser = configparser.ConfigParser()
        if config_file_path is None:
            config_file_path = self.DEFAULT_CONFIG_FILE_PATH

        if not config_file_path.exists():
            raise Exception("Config file doesn't exist! path = " + config_file_path)
        else:
            self.configParser.read(config_file_path)

    def r_home(self):
        return self.configParser["active"]["R_HOME"]

    def matlab_path(self):
        return self.configParser["active"]["MATLAB_PATH"]
