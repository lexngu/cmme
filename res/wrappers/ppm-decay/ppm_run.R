# https://rpy2.github.io/doc/v3.4.x/html/robjects_rpackages.html#importing-arbitrary-r-code-as-a-package
library(ppm)
library(dplyr, warn.conflicts=FALSE)
library(tidyr)

run_ppm <- function(instructions_file_path) {
  # read instruction file
  params <- read.csv(instructions_file_path)
  
  # Run SIMPLE or DECAY?
  requested_model <- params$model

  # map data types
  if (requested_model == "DECAY") {
    params$only_learn_from_buffer <- as.logical(params$only_learn_from_buffer)
    params$only_predict_from_buffer <- as.logical(params$only_predict_from_buffer)
  } else if (requested_model == "SIMPLE") {
    params$shortest_deterministic <- as.logical(params$shortest_deterministic)
    params$exclusion <- as.logical(params$exclusion)
    params$update_exclusion <- as.logical(params$update_exclusion)
  }

  # parse alphabet_levels
  alphabet_levels <- strsplit(gsub("\\[(.*?)\\]", "\\1", params$alphabet_levels), ",\\s*")[[1]]
  if (length(alphabet_levels) == 0) {
    alphabet_levels <- character()
  }
  
  # parse input_sequence
  input_sequence <- strsplit(gsub("\\[(.*?)\\]", "\\1", params$input_sequence), ",\\s*")[[1]]
  if (length(alphabet_levels) > 0) {
    input_sequence <- factor(input_sequence, levels = alphabet_levels)
  } else {
    input_sequence <- as.factor(input_sequence)
  }
  
  # parse input_time_seq (if DECAY)
  if (requested_model == "DECAY") {
    input_time_seq <- strsplit(gsub("\\[(.*?)\\]", "\\1", params$input_time_sequence), ",\\s*")[[1]]
    input_time_seq <- as.numeric(input_time_seq)
  }
  
  # run model
  if (requested_model == "DECAY") {
    mod <- new_ppm_decay(params$alphabet_size, 
                         order_bound = params$order_bound, 
                         ltm_weight = params$ltm_weight, ltm_half_life = params$ltm_half_life, ltm_asymptote = params$ltm_asymptote, 
                         noise = params$noise, 
                         stm_weight = params$stm_weight, stm_duration = params$stm_duration, 
                         buffer_weight = params$buffer_weight, buffer_length_time = params$buffer_length_time, 
                         buffer_length_items = params$buffer_length_items, 
                         only_learn_from_buffer = params$only_learn_from_buffer, 
                         only_predict_from_buffer = params$only_predict_from_buffer, 
                         seed = params$seed, 
                         alphabet_levels = alphabet_levels
                         )
    # raw results
    res <- model_seq(mod, input_sequence, time=input_time_seq)
  } else if (requested_model == "SIMPLE") {
    mod <- new_ppm_simple(params$alphabet_size, 
                          order_bound = params$order_bound,
                          shortest_deterministic = params$shortest_deterministic,
                          exclusion = params$exclusion,
                          update_exclusion = params$update_exclusion,
                          escape = params$escape,
                          alphabet_levels = alphabet_levels)
    # raw results
    res <- model_seq(mod, input_sequence)
  }

  # prepare results for storage in results file
  ppm_res <- res %>% 
    # expand probability distribution's values to multiple rows
    unnest_longer(distribution, indices_to = "probability_distribution_value_for_alphabet_idx", values_to = "probability_distribution_value") %>% 
    # add matching symbol to probability_distribution_value_for_alphabet_idx
    rowwise() %>% mutate(probability_distribution_value_for_symbol = levels(input_sequence)[probability_distribution_value_for_alphabet_idx]) %>%
    # change column order
    select(pos, time, symbol, probability_distribution_value_for_alphabet_idx, probability_distribution_value_for_symbol, probability_distribution_value, everything())
  
  # write results file
  write.csv(ppm_res, params$results_file_path)
  
  return(params$results_file_path)
}
