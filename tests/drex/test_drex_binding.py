import tempfile

import numpy as np

from cmme.drex.base import UnprocessedPrior, DistributionType
from cmme.drex.binding import from_mat, to_mat, InstructionsFile
from cmme.drex.util import auto_convert_input_sequence, trialtimefeature_sequence_as_singletrial_array, \
    trialtimefeature_sequence_as_multitrial_cell


def prior_input_sequence_from_mat_to_trialtimefeature_sequence(prior_input_sequence_from_mat):
    data = prior_input_sequence_from_mat[0][0]
    trial_count = data.shape[0]
    res = list()
    for trial_idx in range(trial_count):
        res.append(data[trial_idx].T.tolist())
    return auto_convert_input_sequence(res)

def test_to_mat_and_from_mat():
    data = {
        "a": "b",
        "c": [1, 2, 3],
        "d": float('inf')
    }

    tmp_file_path = tempfile.NamedTemporaryFile().name

    to_mat(data, tmp_file_path)
    data_after_save_and_read = from_mat(tmp_file_path)

    assert data["a"] == data_after_save_and_read["a"][0]
    assert data["c"] == data_after_save_and_read["c"][0].tolist()
    assert data["d"] == data_after_save_and_read["d"][0]


def test_drex_instructions_file():
    results_file_path = "abc123.mat"
    input_sequence = auto_convert_input_sequence([1, 2, 3, 4, 5])
    prior_input_sequence = auto_convert_input_sequence([[[11, 12, 13, 14, 15]], [[21, 22, 23, 24, 25]]])
    prior = UnprocessedPrior(DistributionType.GMM, prior_input_sequence)
    hazard = 0.1
    memory = float('inf')
    maxhyp = 2
    obsnz = 0.1
    change_decision_threshold = 0.1
    instructions_file = InstructionsFile(results_file_path, input_sequence, prior, hazard, memory, maxhyp, obsnz, change_decision_threshold)

    tmp_file_path = tempfile.NamedTemporaryFile().name

    instructions_file.write_to_mat(tmp_file_path)

    data_after_read = from_mat(tmp_file_path)
    assert instructions_file.results_file_path == data_after_read["results_file_path"][0]
    assert np.array_equal(trialtimefeature_sequence_as_multitrial_cell(prior_input_sequence), trialtimefeature_sequence_as_multitrial_cell(prior_input_sequence_from_mat_to_trialtimefeature_sequence(data_after_read["estimate_suffstat"]["xs"][0])))
    assert prior.distribution_type().value == data_after_read["estimate_suffstat"]["params"][0][0][0]["distribution"][0][0]
    assert prior.D_value() == data_after_read["estimate_suffstat"]["params"][0][0][0]["D"][0][0][0]
    assert prior.max_n_comp == data_after_read["estimate_suffstat"]["params"][0][0][0]["max_ncomp"][0][0][0]
    assert np.array_equal(trialtimefeature_sequence_as_singletrial_array(instructions_file.input_sequence), data_after_read["run_DREX_model"]["x"][0][0])
    assert prior.distribution_type().value == data_after_read["run_DREX_model"]["params"][0][0][0]["distribution"][0][0]
    assert prior.D_value() == data_after_read["run_DREX_model"]["params"][0][0][0]["D"][0][0][0]
    assert instructions_file.hazard == data_after_read["run_DREX_model"]["params"][0][0][0]["hazard"][0][0][0]
    assert instructions_file.memory == data_after_read["run_DREX_model"]["params"][0][0][0]["memory"][0][0][0]
    assert instructions_file.maxhyp == data_after_read["run_DREX_model"]["params"][0][0][0]["maxhyp"][0][0][0]
    assert instructions_file.obsnz == data_after_read["run_DREX_model"]["params"][0][0][0]["obsnz"][0][0][0]
    assert prior.max_n_comp == data_after_read["run_DREX_model"]["params"][0][0][0]["max_ncomp"][0][0][0]
    assert instructions_file.change_decision_threshold == data_after_read["post_DREX_changedecision"]["threshold"][0][0][0]

