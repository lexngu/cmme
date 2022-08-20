# CMME
library(ppm)
library(dplyr, warn.conflicts=FALSE)
library(tidyr)

# Reads the instructions file at +instructions_file_path+, runs the model, writes the results file and returns the results file path.
run_ppm <- function(instructions_file_path) {
  ####
  # 1. Read instructions file (.csv)
  params <- read.csv(instructions_file_path)
  ####
  
  ####
  # 2. Prepare instantiation
  ####
  # Extract model_type \in {SIMPLE, DECAY}
  model_type <- params$model

  # Do data type conversions
  if (model_type == "DECAY") {
    params$only_learn_from_buffer <- as.logical(params$only_learn_from_buffer)
    params$only_predict_from_buffer <- as.logical(params$only_predict_from_buffer)
  } else if (model_type == "SIMPLE") {
    params$shortest_deterministic <- as.logical(params$shortest_deterministic)
    params$exclusion <- as.logical(params$exclusion)
    params$update_exclusion <- as.logical(params$update_exclusion)
  }

  # Parse alphabet_levels, therefore split string using the separator ", "
  alphabet_levels <- strsplit(params$alphabet_levels, ", ")[[1]]
  
  # Parse input_time_seq (if DECAY)
  if (model_type == "DECAY") {
    input_time_seq <- strsplit(params$input_time_sequence, ", ")[[1]]
    input_time_seq <- as.numeric(input_time_seq)
  }
  
  # Parse input_sequence
  input_sequence <- strsplit(params$input_sequence, ", ")[[1]]
  input_sequence <- factor(input_sequence, levels = alphabet_levels)
  
  ####
  # 3. Run model
  ####
  # Run model
  if (model_type == "DECAY") {
    mod <- new_ppm_decay(alphabet_levels = alphabet_levels,
                         
                         order_bound = params$order_bound, 
                         
                         buffer_weight = params$buffer_weight, buffer_length_time = params$buffer_length_time, buffer_length_items = params$buffer_length_items, 
                         only_learn_from_buffer = params$only_learn_from_buffer, 
                         only_predict_from_buffer = params$only_predict_from_buffer, 
                         stm_weight = params$stm_weight, stm_duration = params$stm_duration, 
                         ltm_weight = params$ltm_weight, ltm_half_life = params$ltm_half_life, ltm_asymptote = params$ltm_asymptote, 
                         noise = params$noise, 
                         seed = params$seed
                         )
    res <- model_seq(mod, input_sequence, time=input_time_seq)
  } else if (model_type == "SIMPLE") {
    mod <- new_ppm_simple(alphabet_levels = alphabet_levels,
                          
                          order_bound = params$order_bound,
                          
                          shortest_deterministic = params$shortest_deterministic,
                          exclusion = params$exclusion,
                          update_exclusion = params$update_exclusion,
                          escape = params$escape
                          )
    res <- model_seq(mod, input_sequence)
  }

  ####
  # 4. Write results file
  ####
  # 4.1 Results file
  ppm_res <- res %>% 
    # Expand probability distribution's values to multiple rows
    unnest_longer(distribution, indices_to = "probability_distribution_value_for_alphabet_idx", values_to = "probability_distribution_value") %>% 
    # Add matching symbol to probability_distribution_value_for_alphabet_idx
    rowwise() %>% mutate(probability_distribution_value_for_symbol = levels(input_sequence)[probability_distribution_value_for_alphabet_idx]) %>%
    # Change column order
    select(pos, time, symbol, probability_distribution_value_for_alphabet_idx, probability_distribution_value_for_symbol, probability_distribution_value, everything())

  # Write results file
  write.csv(ppm_res, params$results_file_path)
  
  # 4.2 Meta results file
  meta_information = df <- data.frame(
    instructions_file_path=character(),
    results_file_path=character()
  )
  
  meta_information[1, ] = c(
    instructions_file_path,
    params$results_file_path
  )
  write.csv(meta_information, gsub("\\.csv", "-meta.csv", params$results_file_path))
  
  # Return results_file_path
  return(params$results_file_path)
}
