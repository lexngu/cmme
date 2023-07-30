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

    assert model == IDYOMModelValue.BOTH_PLUS
    assert stm_options['order_bound'] == None
    assert stm_options['mixtures'] == True
    assert stm_options['update_exclusion'] == True
    assert stm_options['escape'] == IDYOMEscape.X
    assert ltm_options['order_bound'] == None
    assert ltm_options['mixtures'] == True
    assert ltm_options['update_exclusion'] == False
    assert ltm_options['escape'] == IDYOMEscape.C
    if len(training_options.keys()) > 0:
        assert training_options['resampling_folds_count'] == 10
        assert training_options['exclusively_to_be_used_resampling_fold_indices'] == None
    if len(select_options.keys()) > 0:
        assert select_options['dp'] == 2
        assert select_options['max_links'] == 2
        assert select_options['min_links'] == 2
    assert output_options['overwrite'] == False
    assert output_options['separator'] == " "


def test_idyom_run_succeeds():
    with tempfile.TemporaryDirectory() as tmpdir:
        install_idyom(tmpdir)
        idyom_model = IDYOMModel(Config().idyom_root_path(), Path(tmpdir) / Path("./db/database.sqlite"))

        midi_dir_path = str(Path(__file__).parent.parent.resolve() / Path("sample_files/idyom-midi")) + "/"
        dataset = idyom_model.idyom_database.import_midi_dataset(midi_dir_path, "test")

        idyomib = IDYOMInstructionBuilder()
        idyomib.source_viewpoints([BasicViewpoint.CPITCH]).target_viewpoints([BasicViewpoint.CPITCH]).dataset(dataset)

        idyom_results_file = idyom_model.run(idyomib)

        assert idyom_results_file.df is not None
