from cmme.config import Config
from cmme.idyom import IDYOMDatabase
from cmme.idyom.model import *
from cmme.idyom.util import install_idyom


def test_default_idyom_instruction_builder_uses_default_values():
    idyomib = IDYOMInstructionBuilder()

    model = idyomib._model
    stm_options = idyomib._stm_options
    ltm_options = idyomib._ltm_options
    training_options = idyomib._training_options
    select_options = idyomib._select_options
    output_options = idyomib._output_options

    assert model == IDYOMModelType.BOTH_PLUS
    assert stm_options['order_bound'] is None
    assert stm_options['mixtures'] is None
    assert stm_options['update_exclusion'] is None
    assert stm_options['escape'] is None
    assert ltm_options['order_bound'] is None
    assert ltm_options['mixtures'] is None
    assert ltm_options['update_exclusion'] is None
    assert ltm_options['escape'] is None
    if len(training_options.keys()) > 0:
        assert training_options['resampling_folds_count'] is None
        assert training_options['exclusively_to_be_used_resampling_fold_indices'] is None
    if len(select_options.keys()) > 0:
        assert select_options['dp'] is None
        assert select_options['max_links'] is None
        assert select_options['min_links'] is None
    assert output_options['overwrite'] is None
    assert output_options['separator'] is None


def test_idyom_stm_run_succeeds():
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_root_path = Config().idyom_root_path()
        idyom_database_path = Path(tmpdir) / Path("./db/database.sqlite")
        install_idyom(idyom_root_path, idyom_database_path)

        midi_dir_path = str(Path(__file__).parent.parent.resolve() / Path("sample_files/idyom-midi")) + "/"
        idyom_database = IDYOMDatabase(idyom_root_path, idyom_database_path)
        dataset = idyom_database.import_midi_dataset(midi_dir_path, "test")

        idyomib = IDYOMInstructionBuilder()
        idyomib.model(IDYOMModelType.STM).source_viewpoints([BasicViewpoint.CPITCH])\
            .target_viewpoints([BasicViewpoint.CPITCH]).dataset(dataset)\
            .idyom_root_path(idyom_root_path).idyom_database_path(idyom_database_path)\
            .training_options(resampling_folds_count_k=1)

        idyomif = idyomib.to_instructions_file()

        idyom_model = IDYOMModel()
        idyom_results_file = idyom_model.run_instructions_file(idyomif)

        assert idyom_results_file.df is not None

def test_idyom_both_run_succeeds():
    with tempfile.TemporaryDirectory() as tmpdir:
        idyom_root_path = Config().idyom_root_path()
        idyom_database_path = Path(tmpdir) / Path("./db/database.sqlite")
        install_idyom(idyom_root_path, idyom_database_path)

        midi_dir_path = str(Path(__file__).parent.parent.resolve() / Path("sample_files/idyom-midi")) + "/"
        idyom_database = IDYOMDatabase(idyom_root_path, idyom_database_path)
        dataset = idyom_database.import_midi_dataset(midi_dir_path, "test")

        idyomib = IDYOMInstructionBuilder()
        idyomib.source_viewpoints([BasicViewpoint.CPITCH])\
            .target_viewpoints([BasicViewpoint.CPITCH])\
            .dataset(dataset)\
            .model(IDYOMModelType.BOTH)\
            .ltm_options(order_bound=2)\
            .idyom_root_path(idyom_root_path).idyom_database_path(idyom_database_path)\
            .training_options(resampling_folds_count_k=1)

        idyomif = idyomib.to_instructions_file()

        idyom_model = IDYOMModel()
        idyom_results_file = idyom_model.run_instructions_file(idyomif)

        assert idyom_results_file.df is not None