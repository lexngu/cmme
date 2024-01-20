import configparser
from pathlib import Path


class Config:
    DEFAULT_CONFIG_FILE_PATH = (Path(__file__).parent / "../../cmme-comparison.ini").resolve()

    # Section to use
    CONFIG_SECTION_KEY = "active"

    # Keys to look for
    CONFIG_R_HOME_KEY = "R_HOME"
    CONFIG_MATLAB_PATH_KEY = "MATLAB_PATH"
    CONFIG_IDYOM_ROOT = "IDYOM_ROOT"
    CONFIG_IDYOM_DATABASE = "IDYOM_DATABASE"
    CONFIG_CMME_IO_DIR_KEY = "CMME_IO_DIR"

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

    def idyom_root_path(self) -> Path:
        return Path(self.config_parser[Config.CONFIG_SECTION_KEY][Config.CONFIG_IDYOM_ROOT])

    def idyom_database_path(self) -> Path:
        return Path(self.config_parser[Config.CONFIG_SECTION_KEY][Config.CONFIG_IDYOM_DATABASE])

    def cmme_io_dir(self) -> Path:
        return Path(self.config_parser[Config.CONFIG_SECTION_KEY][Config.CONFIG_CMME_IO_DIR_KEY])