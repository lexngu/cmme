from cmme.drex.distribution.base import DistributionType
from cmme.drex.distribution.prior import UnprocessedPrior
from cmme.drex.model import *


def test_default_drex_instance_uses_original_default_values():
    prior_distribution_type = DistributionType.GAUSSIAN
    prior_input_sequence = [2, 3, 4]
    prior_D = 2
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D)
    drex_model = DREXModel(prior)
    drex_instance = drex_model.instance

    assert drex_instance._hazard == 0.01
    assert drex_instance._memory == np.inf
    assert drex_instance._maxhyp == np.inf
    assert drex_instance._change_decision_threshold == 0.01
    assert drex_instance._obsnz == 0

def test_to_instructions_file_uses_specified_values():
    prior_distribution_type = DistributionType.GMM
    prior_input_sequence = [2, 3, 4]
    prior_D = 1
    prior_max_n_comp = 9
    prior_beta = 0.02
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D,
                                                 prior_max_n_comp, prior_beta)
    drex_model = DREXModel(prior)
    drex_instance = drex_model.instance

    instructions_file_path = "instructions_file.mat"
    results_file_path = "results_file.mat"

    input_sequence = auto_convert_input_sequence([1,2,3])
    hazard = 0.12
    memory = 22
    maxhyp = 11
    obsnz = 0.03
    change_decision_threshold = 0.5
    drex_instance.with_input_sequence(input_sequence)
    drex_instance.with_hazard(hazard)
    drex_instance.with_memory(memory)
    drex_instance.with_maxhyp(maxhyp)
    drex_instance.with_obsnz(obsnz)
    drex_instance.with_change_decision_threshold(change_decision_threshold)

    instructions_file = drex_model.to_instructions_file(instructions_file_path, results_file_path)

    assert instructions_file.instructions_file_path == instructions_file_path
    assert instructions_file.results_file_path == results_file_path
    assert np.array_equal(instructions_file.input_sequence, input_sequence)
    assert instructions_file.prior == prior
    assert instructions_file.hazard == hazard
    assert instructions_file.memory == memory
    assert instructions_file.maxhyp == maxhyp
    assert instructions_file.obsnz == obsnz
    assert instructions_file.change_decision_threshold == change_decision_threshold

def test_run_succeeds():
    prior_distribution_type = DistributionType.GMM
    prior_input_sequence = [1, 1, 1, 1, 4, 4, 4, 4, 1, 2, 3, 4, 2, 2, 2, 2, 1, 1, 1, 1]
    prior_D = 1
    prior_max_n_comp = 9
    prior_beta = 0.02
    prior = UnprocessedPrior(prior_distribution_type, prior_input_sequence, prior_D,
                                                 prior_max_n_comp, prior_beta)
    drex_model = DREXModel(prior)
    drex_instance = drex_model.instance

    input_sequence = auto_convert_input_sequence([1, 2, 3])
    hazard = 0.12
    memory = 22
    maxhyp = 11
    obsnz = 0.03
    change_decision_threshold = 0.5
    drex_instance.with_input_sequence(input_sequence)
    drex_instance.with_hazard(hazard)
    drex_instance.with_memory(memory)
    drex_instance.with_maxhyp(maxhyp)
    drex_instance.with_obsnz(obsnz)
    drex_instance.with_change_decision_threshold(change_decision_threshold)

    instructions_file_path = drex_default_instructions_file_path()
    results_file_path = drex_default_results_file_path()
    results_file = drex_model.run(instructions_file_path, results_file_path)

    assert results_file.instructions_file_path == str(instructions_file_path)
    assert np.array_equal(results_file.input_sequence, input_sequence)
