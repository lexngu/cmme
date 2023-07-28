import tempfile
from pathlib import Path

from cmme.drex.base import UnprocessedPrior, DistributionType
from cmme.drex.model import *
from cmme.drex.binding import transform_to_rundrexmodel_representation
from cmme.drex.util import transform_to_unified_drex_input_sequence_representation
from cmme.lib.util import drex_default_instructions_file_path, drex_default_results_file_path
from cmme.drex.worker import DREXModel


def test_default_drex_instance_uses_original_default_values():
    drex_instance = DREXInstructionBuilder()

    assert drex_instance._hazard == 0.01
    assert drex_instance._memory == np.inf
    assert drex_instance._maxhyp == np.inf
    assert drex_instance._change_decision_threshold == 0.01
    assert drex_instance._obsnz == 0


def test_to_instructions_file_uses_specified_values():
    drex_instance = DREXInstructionBuilder()

    instructions_file_path = "instructions_file.mat"
    results_file_path = "results_file.mat"

    prior_distribution_type = DistributionType.GMM
    prior_input_sequence = [2, 3, 4]
    prior_D = 1
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D)
    input_sequence = transform_to_unified_drex_input_sequence_representation([1, 2, 3])
    hazard = 0.12
    memory = 22
    maxhyp = 11
    max_ncomp = 9
    beta = 0.02
    predscale = 0.987
    obsnz = [0.03]
    change_decision_threshold = 0.5
    drex_instance.input_sequence(input_sequence)
    drex_instance.hazard(hazard)
    drex_instance.memory(memory)
    drex_instance.maxhyp(maxhyp)
    drex_instance.obsnz(obsnz)
    drex_instance.change_decision_threshold(change_decision_threshold)
    drex_instance.prior(prior)
    drex_instance.max_ncomp(max_ncomp)
    drex_instance.beta(beta)
    drex_instance.predscale(predscale)

    instructions_file = drex_instance.to_instructions_file()

    assert np.array_equal(instructions_file.input_sequence, input_sequence)
    assert instructions_file.prior == prior
    assert instructions_file.hazard == hazard
    assert instructions_file.memory == memory
    assert instructions_file.maxhyp == maxhyp
    assert instructions_file.obsnz == obsnz
    assert instructions_file.predscale == predscale
    assert instructions_file.max_ncomp == max_ncomp
    assert instructions_file.beta == beta
    assert instructions_file.change_decision_threshold == change_decision_threshold


def test_run_succeeds():
    prior_distribution_type = DistributionType.GMM
    prior_input_sequence = [1, 1, 1, 1, 4, 4, 4, 4, 1, 2, 3, 4, 2, 2, 2, 2, 1, 1, 1, 1]
    prior_D = 1
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D)
    drex_instance = DREXInstructionBuilder()

    input_sequence = transform_to_unified_drex_input_sequence_representation([1, 2, 3])
    hazard = 0.12
    memory = 22
    maxhyp = 11
    obsnz = 0.03
    change_decision_threshold = 0.5
    drex_instance.input_sequence(input_sequence)
    drex_instance.hazard(hazard)
    drex_instance.memory(memory)
    drex_instance.maxhyp(maxhyp)
    drex_instance.obsnz(obsnz)
    drex_instance.change_decision_threshold(change_decision_threshold)
    drex_instance.prior(prior)
    with tempfile.TemporaryDirectory() as tmpdirname:
        instructions_file_path = drex_default_instructions_file_path(None, Path(tmpdirname))
        results_file_path = drex_default_results_file_path(None, Path(tmpdirname))

        drex_model = DREXModel()
        drex_instance.to_instructions_file()\
            .save_self(instructions_file_path, results_file_path)
        results_file = drex_model.run(instructions_file_path)

        assert results_file.instructions_file_path == str(instructions_file_path)
        assert np.array_equal(results_file.input_sequence, transform_to_rundrexmodel_representation(input_sequence))


def test_drex_instance_automatically_sets_obsnz_according_to_nfeatures():
    prior_input_sequence = [[[1, 1, 1], [2, 2, 2]]]

    drex_instance = DREXInstructionBuilder()

    drex_instance.obsnz(0)
    drex_instance.input_sequence(prior_input_sequence)

    assert len(drex_instance._obsnz) == 2
