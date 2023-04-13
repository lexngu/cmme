import configparser
from pathlib import Path


class Config:
    DEFAULT_CONFIG_FILE_PATH = (Path(__file__).parent / "../../cmme-comparison.ini").resolve()

    # Section to use
    CONFIG_SECTION_KEY = "active"

    # Keys to look for
    CONFIG_R_HOME_KEY = "R_HOME"
    CONFIG_MATLAB_PATH_KEY = "MATLAB_PATH"
    CONFIG_MODEL_IO_PATH_KEY = "MODEL_IO_PATH"
    CONFIG_IDYOM_ROOT = "IDYOM_ROOT"
    CONFIG_IDYOM_DATABASE = "IDYOM_DATABASE"

    def __init__(self, config_file_path: Path = DEFAULT_CONFIG_FILE_PATH):
        if not config_file_path.exists():
            raise Exception("Config file doesn't exist! path = " + str(config_file_path))

        self.config_file_path = config_file_path
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(config_file_path)

    def r_home(self) -> Path:
        return Path(self.config_parser[Config.CONFIG_SECTION_KEY][Config.CONFIG_R_HOME_KEY])

    def matlab_path(self) -> Path:
        return Path(self.config_parser[Config.CONFIG_SECTION_KEY][Config.CONFIG_MATLAB_PATH_KEY])

    def model_io_path(self) -> Path:
        return Path(self.config_file_path.parent / self.config_parser[Config.CONFIG_SECTION_KEY][Config.CONFIG_MODEL_IO_PATH_KEY]).resolve()

    def idyom_root_path(self) -> Path:
        return Path(self.config_parser[Config.CONFIG_SECTION_KEY][Config.CONFIG_IDYOM_ROOT])

    def idyom_database_path(self) -> Path:
        return Path(self.config_parser[Config.CONFIG_SECTION_KEY][Config.CONFIG_IDYOM_DATABASE])