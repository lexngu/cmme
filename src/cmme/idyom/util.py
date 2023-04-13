from pathlib import Path
import os

def path_with_trailing_slash(path) -> Path:
    """
    Takes a Path or any object that can be converted into a Path object and returns a Path for which a trailing slash is ensured.
    :param path:
    :return:
    """
    if not isinstance(path, Path):
        path = Path(path)
    return Path(os.path.join(path, ''))