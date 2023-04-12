from cmme.ppmdecay.model import *
from cmme.ppmdecay.util import str_to_list, list_to_str


def test_default_ppm_simple_instance_uses_original_default_values():
    ppmsimple_instance = PPMSimpleInstance()

    assert ppmsimple_instance._order_bound == 10
    assert ppmsimple_instance._shortest_deterministic == True
    assert ppmsimple_instance._exclusion == True
    assert ppmsimple_instance._update_exclusion == True
    assert ppmsimple_instance._escape_method == EscapeMethod.C

def test_default_ppm_decay_instance_uses_original_default_values():
    ppmdecay_instance = PPMDecayInstance()

    assert ppmdecay_instance._order_bound == 10
    assert ppmdecay_instance._buffer_weight == 1
    assert ppmdecay_instance._buffer_length_time == 0
    assert ppmdecay_instance._buffer_length_items == 0
    assert ppmdecay_instance._only_learn_from_buffer == False
    assert ppmdecay_instance._only_predict_from_buffer == False
    assert ppmdecay_instance._stm_weight == 1
    assert ppmdecay_instance._stm_duration == 0
    assert ppmdecay_instance._ltm_weight == 1
    assert ppmdecay_instance._ltm_half_life == 10
    assert ppmdecay_instance._ltm_asymptote == 0
    assert ppmdecay_instance._noise == 0

def test_run_ppm_simple_succeeds():
    alphabet_levels = [1,2,3,4,5,6]
    order_bound = 2
    input_sequence = [1,2,3,4,5]
    escape_method = EscapeMethod.B
    shortest_deterministic = False
    update_exclusion = False
    exclusion = False
    instructions_file_path = ppmdecay_default_instructions_file_path("test")
    results_file_path = ppmdecay_default_results_file_path("test")

    ppmsimple_instance = PPMSimpleInstance()
    ppmsimple_instance.alphabet_levels(alphabet_levels)\
        .order_bound(order_bound).input_sequence(input_sequence)\
        .escape_method(escape_method).shortest_deterministic(shortest_deterministic).update_exclusion(update_exclusion).exclusion(exclusion)

    ppm_model = PPMModel(ppmsimple_instance)
    results_meta_file = ppm_model.run(instructions_file_path, results_file_path)

    assert results_meta_file._model_type == ModelType.SIMPLE
    assert results_meta_file._alphabet_levels == str_to_list(list_to_str(alphabet_levels))
    assert results_meta_file._instructions_file_path == str(instructions_file_path)



def test_run_ppm_decay_succeeds():
    alphabet_levels = [1, 2, 3, 4, 5, 6]
    order_bound = 1
    input_sequence = [1, 2, 3, 4, 5]
    input_time_sequence = [1, 3, 4.5, 5, 10]
    buffer_weight = 8
    buffer_length_time = 2
    buffer_length_items = 3
    only_learn_from_buffer = True
    only_predict_from_buffer = True
    stm_weight = 7
    stm_duration = 1
    ltm_weight = 6
    ltm_half_life = 1.5
    ltm_asymptote = 0.1
    noise = 0.05
    seed = 999
    instructions_file_path = ppmdecay_default_instructions_file_path("test")
    results_file_path = ppmdecay_default_results_file_path("test")

    ppmdecay_instance = PPMDecayInstance()
    ppmdecay_instance.alphabet_levels(alphabet_levels) \
        .order_bound(order_bound).input_sequence(input_sequence).input_time_sequence(input_time_sequence)\
        .buffer_weight(buffer_weight).buffer_length_time(buffer_length_time).buffer_length_items(buffer_length_items)\
        .only_learn_from_buffer(only_learn_from_buffer).only_predict_from_buffer(only_predict_from_buffer)\
        .stm_weight(stm_weight).stm_duration(stm_duration)\
        .ltm_weight(ltm_weight).ltm_half_life(ltm_half_life).ltm_asymptote(ltm_asymptote)\
        .noise(noise).seed(seed)

    ppm_model = PPMModel(ppmdecay_instance)
    results_meta_file = ppm_model.run(instructions_file_path, results_file_path)

    assert results_meta_file._model_type == ModelType.DECAY
    assert results_meta_file._alphabet_levels == str_to_list(list_to_str(alphabet_levels))
    assert results_meta_file._instructions_file_path == str(instructions_file_path)