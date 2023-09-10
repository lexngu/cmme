import tempfile
import numpy as np
import os

from cmme.config import Config
from cmme.idyom import IdyomDatabase
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

        idb = IdyomDatabase(idyom_root_path, idyom_database_path)

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

        idb = IdyomDatabase(idyom_root_path, idyom_database_path)

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

        idb = IdyomDatabase(idyom_root_path, idyom_database_path)

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

        idb = IdyomDatabase(idyom_root_path, idyom_database_path)

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

        idb = IdyomDatabase(idyom_root_path, idyom_database_path)

        description = "test"
        dataset_id = idb.import_midi_dataset(sample_midi_files_dir_path, description=description)

        compositions = idb.get_all_compositions(dataset_id)
        assert len(compositions) == 1

def test_encode_composition():
    idyom_root_path = Config().idyom_root_path()
    sample_midi_files_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample_files/idyom-midi/"))
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_database_path = path_as_string_with_trailing_slash(tmpdir) + "db/database.sqlite"

        install_idyom(idyom_root_path, idyom_database_path)

        idb = IdyomDatabase(idyom_root_path, idyom_database_path)

        description = "test"
        dataset_id = idb.import_midi_dataset(sample_midi_files_dir_path, description=description)
        compositions = idb.get_all_compositions(dataset_id)
        composition = compositions[0]

        encoding = idb.encode_composition(composition, BasicViewpoint.CPITCH)
        assert encoding == [97, 77, 42, 5, 81, 102, 67, 100, 123, 17, 28, 26, 24, 39, 89, 54, 43, 79, 118, 4, 25, 30, 68, 32, 36, 14, 126, 103, 105, 71, 86, 127, 95, 85, 10, 90, 82, 53, 46, 59, 94, 12, 58, 31, 88, 37, 51, 91, 104, 1, 110, 18, 75, 56, 63, 2, 11, 106, 44, 87, 83, 116, 84, 16, 62, 19, 27, 117, 92, 34, 93, 55, 57, 124, 73, 47, 120, 21, 66, 33, 64, 99, 20, 112, 74, 8, 38, 15, 119, 29, 52, 41, 3, 76, 45, 125, 107, 7, 40, 60, 108, 72, 109, 98, 61, 23, 35, 101, 0, 121, 111, 9, 6, 50, 48, 96, 13, 65, 113, 80, 69, 78, 115, 49, 22, 70, 114, 122]