from pathlib import Path
import datetime
import os


class ModelIO:
    """This class generates filesystem paths for organized file management. Generated paths are used by other
    modules. """

    PPM_PATH = "./ppm/"
    DREX_PATH = "./drex/"
    PLOT_PATH = "./plot/"

    def __init__(self, base_path, alias):
        """
        Initializes the object. Neccessary directories will be created automatically.

        Parameters:
        base_path (str): Base path used to prepend all generated paths.
        alias (str): An alias that helps identifying the generated paths. It will be included in the generated paths.
        """
        self._base_path = Path(base_path).expanduser().resolve()
        self._alias = alias

        self._ppm_base_path = self._base_path / ModelIO.PPM_PATH
        self._drex_base_path = self._base_path / ModelIO.DREX_PATH
        self._plot_base_path = self._base_path / ModelIO.PLOT_PATH

        self._ppm_base_path.mkdir(parents=True, exist_ok=True)
        self._drex_base_path.mkdir(parents=True, exist_ok=True)
        self._plot_base_path.mkdir(parents=True, exist_ok=True)

    def _generate_paths(self, base_path):
        datetime_str = datetime.datetime.now().replace(microsecond=0).isoformat().replace(":", "").replace("-", "")
        generic_filename_prefix = str((base_path / ("./" + datetime_str + "_" + self._alias + "_")).resolve())
        input_file_path = generic_filename_prefix + "input"
        output_file_path = generic_filename_prefix + "output"

        return {
            "base_directory": str(base_path) + os.sep,
            "generic_filename_prefix": generic_filename_prefix,
            "input_file_path": input_file_path,
            "output_file_path": output_file_path
        }

    def generate_ppm_paths(self):
        return self._generate_paths(self._ppm_base_path)

    def generate_drex_paths(self):
        return self._generate_paths(self._drex_base_path)

    def generate_plot_paths(self):
        return self._generate_paths(self._plot_base_path)
