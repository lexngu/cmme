import tempfile
from pathlib import Path

from cmme.idyom.base import IDYOMModelValue, BasicViewpoint, transform_string_list_to_viewpoints_list, IDYOMEscape
from cmme.idyom.binding import IDYOMInstructionsFile
from cmme.idyom.model import IDYOMInstructionBuilder
from cmme.idyom.util import install_idyom
from cmme.lib.util import path_as_string_with_trailing_slash


def test_install_idyom():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        install_idyom(tmpdir_path) # first install
        database_sqlite_path = tmpdir_path / Path("db/database.sqlite")
        assert database_sqlite_path.exists() == True

def test_idyom_instructions_file():
    idyom_root_path = "/tmp/test123"
    idyom_database_path = "/tmp/test123/db/database.sqlite"
    dataset = 1
    model = IDYOMModelValue.BOTH_PLUS
    source_viewpoints = "cpitch"
    target_viewpoints = "cpitch"
    stm_ob = 3
    ltm_ob = 4
    pretraining_ids = [2, 5, 6]
    output_path = "/tmp/results"
    idyom_ib = IDYOMInstructionBuilder()\
        .idyom_root_path(idyom_root_path)\
        .idyom_database_path(idyom_database_path)\
        .dataset(dataset)\
        .model(model)\
        .source_viewpoints(source_viewpoints)\
        .target_viewpoints(target_viewpoints)\
        .stm_options(stm_ob)\
        .ltm_options(ltm_ob)\
        .training_options(pretraining_ids)\
        .output_options(output_path)

    idyom_if = idyom_ib.to_instructions_file()
    assert idyom_if.idyom_root_path == idyom_root_path
    assert idyom_if.idyom_database_path == idyom_database_path
    assert idyom_if.model == model
    assert idyom_if.source_viewpoints == [BasicViewpoint(source_viewpoints)]
    assert idyom_if.target_viewpoints == [BasicViewpoint(source_viewpoints)]
    assert idyom_if.stm_options["order_bound"] == stm_ob
    assert idyom_if.ltm_options["order_bound"] == ltm_ob
    assert idyom_if.training_options["pretraining_dataset_ids"] == pretraining_ids
    assert idyom_if.output_options["output_path"] == output_path

def test_load_idyom_instructions_file_minimal():
    dataset = 1
    source_viewpoints = "cpitch"
    target_viewpoints = "cpitch"
    output_path = path_as_string_with_trailing_slash(tempfile.TemporaryDirectory().name)
    output_options = {
        "output_path": output_path,
        "detail": None,
        "overwrite": None,
        "separator": None
    }
    idyom_ib = IDYOMInstructionBuilder()\
        .dataset(dataset)\
        .source_viewpoints(source_viewpoints)\
        .target_viewpoints(target_viewpoints)\
        .output_options(**output_options)

    with tempfile.NamedTemporaryFile() as tmpfile:
        idyom_if = idyom_ib.to_instructions_file()
        idyom_if.save_self(tmpfile.name)

        test_idyom_if = IDYOMInstructionsFile.load(tmpfile.name)
        assert test_idyom_if.dataset == dataset
        assert test_idyom_if.source_viewpoints == transform_string_list_to_viewpoints_list(source_viewpoints)
        assert test_idyom_if.target_viewpoints == transform_string_list_to_viewpoints_list(target_viewpoints)
        assert test_idyom_if.output_options == output_options

def test_load_idyom_instructions_file_with_more_parameters():
    idyom_root_path = "/tmp/test123"
    idyom_database_path = "/tmp/test123/db/database.sqlite"
    dataset = 1
    model = IDYOMModelValue.BOTH_PLUS
    source_viewpoints = "cpitch"
    target_viewpoints = "cpitch"
    stm_options = {
        "order_bound": 3,
        "mixtures": False,
        "update_exclusion": None,
        "escape": IDYOMEscape.A,
    }
    ltm_options = {
        "order_bound": 4,
        "mixtures": None,
        "update_exclusion": True,
        "escape": None
    }
    training_options = {
        "pretraining_dataset_ids": [2, 5, 6],
        "resampling_folds_count_k": 5,
        "exclusively_to_be_used_resampling_fold_indices": [1, 2, 3]
    }
    output_path = path_as_string_with_trailing_slash(tempfile.TemporaryDirectory().name)
    output_options = {
        "output_path": output_path,
        "detail": 3,
        "overwrite": True,
        "separator": " "
    }
    idyom_ib = IDYOMInstructionBuilder()\
        .idyom_root_path(idyom_root_path)\
        .idyom_database_path(idyom_database_path)\
        .dataset(dataset)\
        .source_viewpoints(source_viewpoints)\
        .target_viewpoints(target_viewpoints)\
        .model(model)\
        .stm_options(**stm_options)\
        .ltm_options(**ltm_options)\
        .training_options(**training_options)\
        .output_options(**output_options)

    with tempfile.NamedTemporaryFile() as tmpfile:
        idyom_if = idyom_ib.to_instructions_file()
        idyom_if.save_self(tmpfile.name)

        test_idyom_if = IDYOMInstructionsFile.load(tmpfile.name)
        assert test_idyom_if.idyom_root_path == idyom_root_path
        assert test_idyom_if.idyom_database_path == idyom_database_path
        assert test_idyom_if.dataset == dataset
        assert test_idyom_if.source_viewpoints == transform_string_list_to_viewpoints_list(source_viewpoints)
        assert test_idyom_if.target_viewpoints == transform_string_list_to_viewpoints_list(target_viewpoints)
        assert test_idyom_if.model == IDYOMModelValue(model).value
        assert test_idyom_if.stm_options == stm_options
        assert test_idyom_if.ltm_options == ltm_options
        assert test_idyom_if.training_options == training_options
        assert test_idyom_if.output_options == output_options
