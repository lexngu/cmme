from pathlib import Path
import datetime
from cmme.lib.util import Config


def new_filepath(alias: str=None, extension: str=None, unique_paths: bool=True) -> Path:
    """
    Create a file path using a comparable naming structure.
    If +unique_paths* is True, a suffix will be added automatically to prevent name collision.
    If the parent directory does not exist, it will be created automatically.

    Parameters
    ----------
    alias: str
        Additional alias which is put into the filename
    extension: str
        File extension to use
    Returns
    -------
    Path
        Path to the file
    """
    datetime_str = datetime.datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "-").replace("-", "")

    base_filepath = "-".join(filter(None, [datetime_str, alias]))
    base_filepath_with_extension = ".".join(filter(None, [base_filepath, extension]))
    filepath = Config().cmme_io_dir() / base_filepath_with_extension

    filepath.parent.mkdir(parents=True, exist_ok=True) # Create directories if needed

    # check if file exists
    if filepath.exists() and unique_paths:
        filepath_segments = filepath.name.split("-")
        if len(filepath_segments) <= 2:
            new_suffix = 2
            base_filepath = "-".join([filepath.stem, str(new_suffix)])
            base_filepath_with_extension = ".".join(filter(None, [base_filepath, extension]))
            filepath = Config().cmme_io_dir() / base_filepath_with_extension
        else:
            filepath_last_segment = filepath_segments[-1]
            if filepath_last_segment.isnumeric():
                new_suffix = int(filepath_last_segment)
                while filepath.exists():
                    new_suffix = new_suffix + 1
                    base_filepath = "-".join([filepath.stem, str(new_suffix)])
                    base_filepath_with_extension = ".".join(filter(None, [base_filepath, extension]))
                    filepath = Config().cmme_io_dir() / base_filepath_with_extension
            else:
                new_suffix = 2
                base_filepath = "-".join([filepath.stem, str(new_suffix)])
                base_filepath_with_extension = ".".join(filter(None, [base_filepath, extension]))
                filepath = Config().cmme_io_dir() / base_filepath_with_extension

    return filepath
