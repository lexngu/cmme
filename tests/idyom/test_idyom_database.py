import tempfile
import numpy as np
import os

from cmme.config import Config
from cmme.idyom.idyom_database import IDYOMDatabase
from cmme.idyom.base import BasicViewpoint
from cmme.idyom.util import install_idyom
from cmme.lib.util import path_as_string_with_trailing_slash


def test_get_all_dataset():
    idyom_root_path = Config().idyom_root_path()
    sample_midi_files_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample_files/idyom-midi/"))
    print(sample_midi_files_dir_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_database_path = path_as_string_with_trailing_slash(tmpdir) + "db/database.sqlite"

        install_idyom(idyom_root_path, idyom_database_path)

        idb = IDYOMDatabase(idyom_root_path, idyom_database_path)

        description = "test"
        dataset_id = idb.import_midi_dataset(sample_midi_files_dir_path, description=description)

        all_datasets = idb.get_all_datasets()
        assert len(all_datasets) == 1

def test_import_midi_dataset():
    idyom_root_path = Config().idyom_root_path()
    sample_midi_files_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample_files/idyom-midi/"))
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_database_path = path_as_string_with_trailing_slash(tmpdir) + "db/database.sqlite"

        install_idyom(idyom_root_path, idyom_database_path)

        idb = IDYOMDatabase(idyom_root_path, idyom_database_path)

        description = "test"
        dataset_id = idb.import_midi_dataset(sample_midi_files_dir_path, description=description)

        all_datasets = idb.get_all_datasets()
        assert len(all_datasets) == 1
        assert all_datasets[0].id == dataset_id
        assert all_datasets[0].description == description

def test_import_kern_dataset():
    idyom_root_path = Config().idyom_root_path()
    sample_kern_files_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample_files/idyom-kern/"))
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_database_path = path_as_string_with_trailing_slash(tmpdir) + "db/database.sqlite"

        install_idyom(idyom_root_path, idyom_database_path)

        idb = IDYOMDatabase(idyom_root_path, idyom_database_path)

        description = "test"
        dataset_id = idb.import_kern_dataset(sample_kern_files_dir_path, description=description, timebase=39473280)

        all_datasets = idb.get_all_datasets()
        assert len(all_datasets) == 1
        assert all_datasets[0].id == dataset_id
        assert all_datasets[0].description == description

def test_get_dataset_alphabet():
    idyom_root_path = Config().idyom_root_path()
    sample_midi_files_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample_files/idyom-midi/"))
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_database_path = path_as_string_with_trailing_slash(tmpdir) + "db/database.sqlite"

        install_idyom(idyom_root_path, idyom_database_path)

        idb = IDYOMDatabase(idyom_root_path, idyom_database_path)

        description = "test"
        dataset_id_one = idb.import_midi_dataset(sample_midi_files_dir_path, description=description)
        dataset_id_two = idb.import_midi_dataset(sample_midi_files_dir_path, description=description)

        alphabet = idb.get_dataset_alphabet([dataset_id_one, dataset_id_two], BasicViewpoint.CPITCH)
        assert alphabet == np.arange(0, 128).tolist()

def test_get_all_compositions():
    idyom_root_path = Config().idyom_root_path()
    sample_midi_files_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample_files/idyom-midi/"))
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_database_path = path_as_string_with_trailing_slash(tmpdir) + "db/database.sqlite"

        install_idyom(idyom_root_path, idyom_database_path)

        idb = IDYOMDatabase(idyom_root_path, idyom_database_path)

        description = "test"
        dataset_id = idb.import_midi_dataset(sample_midi_files_dir_path, description=description)

        compositions = idb.get_all_compositions(dataset_id)
        assert len(compositions) == 1

def test_encode_composition():
    idyom_root_path = Config().idyom_root_path()
    sample_midi_files_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample_files/idyom-encoding-test/midi/"))
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_database_path = path_as_string_with_trailing_slash(tmpdir) + "db/database.sqlite"

        install_idyom(idyom_root_path, idyom_database_path)

        idb = IDYOMDatabase(idyom_root_path, idyom_database_path)

        description = "test"
        dataset_id = idb.import_midi_dataset(sample_midi_files_dir_path, description=description)
        compositions = idb.get_all_compositions(dataset_id)
        composition = compositions[0] # assume: file "test.mid"

        encoding = idb.encode_composition(composition, BasicViewpoint.CPITCH)
        assert encoding == [60, 61, 63, 65, 66, 68, 56, 54, 53, 51, 49, 44, 42, 41, 39, 37, 73, 77, 80, 68, 70, 72, 73, 127, 66, 66, 66, 0, 45, 46, 45]