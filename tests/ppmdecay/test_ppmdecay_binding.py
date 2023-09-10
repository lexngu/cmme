import tempfile

from cmme.ppmdecay.base import EscapeMethod, ModelType
from cmme.ppmdecay.binding import PPMSimpleInstructionsFile, PPMDecayInstructionsFile, PPMResultsMetaFile
from cmme.ppmdecay.model import PPMSimpleInstance, PPMDecayInstance, PPMModel
from cmme.ppmdecay.util import auto_convert_input_sequence


def test_ppmsimple_instructions_file():
    alphabet_levels = [1, 2, 3, 5]
    input_sequence = [1, 1, 3, 2, 5, 5, 1, 3]
    order_bound = 3
    update_exclusion = True
    escape_method = EscapeMethod.AX
    exclusion = True
    shortest_deterministic = True
    ppmib = PPMSimpleInstance() \
        .alphabet_levels(alphabet_levels) \
        .input_sequence(input_sequence) \
        .order_bound(order_bound) \
        .update_exclusion(update_exclusion) \
        .escape_method(escape_method) \
        .exclusion(exclusion) \
        .shortest_deterministic(shortest_deterministic)
    ppmif = ppmib.to_instructions_file()

    assert ppmif.alphabet_levels == alphabet_levels
    assert ppmif.input_sequence == auto_convert_input_sequence(input_sequence)
    assert ppmif.order_bound == order_bound
    assert ppmif.update_exclusion == update_exclusion
    assert ppmif.escape_method == escape_method
    assert ppmif.exclusion == exclusion
    assert ppmif.shortest_deterministic == shortest_deterministic


def test_load_ppmsimple_instructions_file():
    alphabet_levels = list(map(str, [1, 2, 3, 5]))
    input_sequence = list(map(str, [1, 1, 3, 2, 5, 5, 1, 3]))
    order_bound = 3
    update_exclusion = True
    escape_method = EscapeMethod.AX
    exclusion = True
    shortest_deterministic = True
    ppmib = PPMSimpleInstance() \
        .alphabet_levels(alphabet_levels) \
        .input_sequence(input_sequence) \
        .order_bound(order_bound) \
        .update_exclusion(update_exclusion) \
        .escape_method(escape_method) \
        .exclusion(exclusion) \
        .shortest_deterministic(shortest_deterministic)
    ppmif = ppmib.to_instructions_file()

    with tempfile.NamedTemporaryFile() as tmpfile:
        ppmif_path = tmpfile.name
        ppmif.save_self(ppmif_path)

        test_ppmif = PPMSimpleInstructionsFile.load(ppmif_path)
        assert test_ppmif.alphabet_levels == alphabet_levels
        assert (test_ppmif.input_sequence == auto_convert_input_sequence(input_sequence))
        assert test_ppmif.order_bound == order_bound
        assert test_ppmif.update_exclusion == update_exclusion
        assert test_ppmif.escape_method == escape_method
        assert test_ppmif.exclusion == exclusion
        assert test_ppmif.shortest_deterministic == shortest_deterministic


def test_ppmdecay_instructions_file():
    alphabet_levels = [1, 2, 3, 5]
    input_sequence = [1, 1, 3, 2, 5, 5, 1, 3]
    input_time_sequence = [1, 2, 4, 5, 6, 8, 9, 10]
    order_bound = 3
    buffer_weight = 2
    buffer_length_time = 2
    buffer_length_items = 3
    only_learn_from_buffer = True
    only_predict_from_buffer = True
    stm_weight = 1.8
    stm_duration = 3
    ltm_weight = 1.5
    ltm_half_life = 4
    ltm_asymptote = 0.1
    noise = 0.1
    seed = 1

    ppmib = PPMDecayInstance() \
        .alphabet_levels(alphabet_levels) \
        .input_sequence(input_sequence, input_time_sequence) \
        .order_bound(order_bound) \
        .buffer_weight(buffer_weight) \
        .buffer_length_items(buffer_length_items) \
        .buffer_length_time(buffer_length_time) \
        .only_learn_from_buffer(only_learn_from_buffer) \
        .only_predict_from_buffer(only_predict_from_buffer) \
        .stm_weight(stm_weight) \
        .stm_duration(stm_duration) \
        .ltm_weight(ltm_weight) \
        .ltm_asymptote(ltm_asymptote) \
        .ltm_half_life(ltm_half_life) \
        .noise(noise) \
        .seed(seed)

    ppmif = ppmib.to_instructions_file()

    assert ppmif.input_sequence == auto_convert_input_sequence(input_sequence)
    assert ppmif.input_time_sequence == auto_convert_input_sequence(input_time_sequence)
    assert ppmif.buffer_weight == buffer_weight
    assert ppmif.buffer_length_time == buffer_length_time
    assert ppmif.buffer_length_items == buffer_length_items
    assert ppmif.only_learn_from_buffer == only_learn_from_buffer
    assert ppmif.only_predict_from_buffer == only_predict_from_buffer
    assert ppmif.stm_weight == stm_weight
    assert ppmif.stm_duration == stm_duration
    assert ppmif.ltm_weight == ltm_weight
    assert ppmif.ltm_half_life == ltm_half_life
    assert ppmif.ltm_asymptote == ltm_asymptote
    assert ppmif.noise == noise
    assert ppmif.seed == seed


def test_load_ppmdecay_instructions_file():
    alphabet_levels = list(map(str, [1, 2, 3, 5]))
    input_sequence = list(map(str, [1, 1, 3, 2, 5, 5, 1, 3]))
    input_time_sequence = [1, 2, 4, 5, 6, 8, 9, 10]
    order_bound = 3
    buffer_weight = 2
    buffer_length_time = 2
    buffer_length_items = 3
    only_learn_from_buffer = True
    only_predict_from_buffer = True
    stm_weight = 1.8
    stm_duration = 3
    ltm_weight = 1.5
    ltm_half_life = 4
    ltm_asymptote = 0.1
    noise = 0.1
    seed = 1

    ppmib = PPMDecayInstance() \
        .alphabet_levels(alphabet_levels) \
        .input_sequence(input_sequence, input_time_sequence) \
        .order_bound(order_bound) \
        .buffer_weight(buffer_weight) \
        .buffer_length_items(buffer_length_items) \
        .buffer_length_time(buffer_length_time) \
        .only_learn_from_buffer(only_learn_from_buffer) \
        .only_predict_from_buffer(only_predict_from_buffer) \
        .stm_weight(stm_weight) \
        .stm_duration(stm_duration) \
        .ltm_weight(ltm_weight) \
        .ltm_asymptote(ltm_asymptote) \
        .ltm_half_life(ltm_half_life) \
        .noise(noise) \
        .seed(seed)

    ppmif = ppmib.to_instructions_file()

    with tempfile.NamedTemporaryFile() as tmpfile:
        ppmif_path = tmpfile.name
        ppmif.save_self(ppmif_path)

        test_ppmif = PPMDecayInstructionsFile.load(ppmif_path)

        assert test_ppmif.input_sequence == auto_convert_input_sequence(input_sequence)
        assert test_ppmif.input_time_sequence == auto_convert_input_sequence(input_time_sequence)
        assert test_ppmif.buffer_weight == buffer_weight
        assert test_ppmif.buffer_length_time == buffer_length_time
        assert test_ppmif.buffer_length_items == buffer_length_items
        assert test_ppmif.only_learn_from_buffer == only_learn_from_buffer
        assert test_ppmif.only_predict_from_buffer == only_predict_from_buffer
        assert test_ppmif.stm_weight == stm_weight
        assert test_ppmif.stm_duration == stm_duration
        assert test_ppmif.ltm_weight == ltm_weight
        assert test_ppmif.ltm_half_life == ltm_half_life
        assert test_ppmif.ltm_asymptote == ltm_asymptote
        assert test_ppmif.noise == noise
        assert test_ppmif.seed == seed


def test_load_ppmdecay_results_file():
    alphabet_levels = list(map(str, [1, 2, 3, 5]))
    input_sequence = list(map(str, [1, 1, 3, 2, 5, 5, 1, 3]))
    input_time_sequence = [1, 2, 4, 5, 6, 8, 9, 10]

    ppmib = PPMDecayInstance() \
        .alphabet_levels(alphabet_levels) \
        .input_sequence(input_sequence, input_time_sequence)

    ppmif = ppmib.to_instructions_file()
    ppmmodel = PPMModel()
    ppmrf = ppmmodel.run_instructions_file(ppmif)
    assert ppmrf.model_type == ModelType.DECAY
    assert ppmrf.alphabet_levels == alphabet_levels

    ppmrf_data = ppmrf.results_file_data
    assert ppmrf_data.df is not None
    assert len(ppmrf_data.trials) == 1


def test_load_ppmsimple_results_file():
    alphabet_levels = list(map(str, [1, 2, 3, 5]))
    input_sequence = list(map(str, [1, 1, 3, 2, 5, 5, 1, 3]))

    ppmib = PPMSimpleInstance() \
        .alphabet_levels(alphabet_levels) \
        .input_sequence(input_sequence)

    ppmif = ppmib.to_instructions_file()
    ppmmodel = PPMModel()
    ppmrf = ppmmodel.run_instructions_file(ppmif)
    assert ppmrf.model_type == ModelType.SIMPLE
    assert ppmrf.alphabet_levels == alphabet_levels

    ppmrf_data = ppmrf.results_file_data
    assert ppmrf_data.df is not None
    assert len(ppmrf_data.trials) == 1
