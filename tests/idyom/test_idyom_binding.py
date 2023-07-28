import os.path
import tempfile
from pathlib import Path

from cmme.idyom.util import install_idyom


def test_install_idyom():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        install_idyom(tmpdir_path) # first install
        database_sqlite_path = tmpdir_path / Path("db/database.sqlite")
        assert database_sqlite_path.exists() == True
