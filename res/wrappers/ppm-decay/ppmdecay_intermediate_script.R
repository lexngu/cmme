library(ppm)
library(arrow, warn.conflicts = FALSE)
ppmdecay_intermediate_script <- function(instructions_file_path) {
  # Read instructions file
  instructions_file <- arrow::read_feather(instructions_file_path)

  if (file.exists(instructions_file$results_file_path)) {
    print(paste("Results file", instructions_file$results_file_path, "already exists."))
    return(instructions_file$results_file_path)
  }
  
  model_type <- instructions_file$model_type # \in {SIMPLE, DECAY}
  # Data type conversions
  if (model_type == "DECAY") {
    instructions_file$only_learn_from_buffer <- as.logical(instructions_file$only_learn_from_buffer)
    instructions_file$only_predict_from_buffer <- as.logical(instructions_file$only_predict_from_buffer)
  } else if (model_type == "SIMPLE") {
    instructions_file$shortest_deterministic <- as.logical(instructions_file$shortest_deterministic)
    instructions_file$exclusion <- as.logical(instructions_file$exclusion)
    instructions_file$update_exclusion <- as.logical(instructions_file$update_exclusion)
  }

  # Parse alphabet_levels, therefore split string using the separator ", "
  alphabet_levels <- strsplit(instructions_file$alphabet_levels, ", ")[[1]]
  
  # Parse input_time_seq (if DECAY)
  if (model_type == "DECAY") {
    input_time_seq <- strsplit(instructions_file$input_time_sequence, ", ")[[1]]
    input_time_seq <- as.numeric(input_time_seq)
  }
  
  # Parse input_sequence
  input_sequence <- strsplit(instructions_file$input_sequence, ", ")[[1]]
  input_sequence <- factor(input_sequence, levels = alphabet_levels)
  
  # Invoke model
  if (model_type == "DECAY") {
    mod <- new_ppm_decay(alphabet_levels = alphabet_levels,

                         order_bound = instructions_file$order_bound, 
                         
                         buffer_weight = instructions_file$buffer_weight, buffer_length_time = instructions_file$buffer_length_time, buffer_length_items = instructions_file$buffer_length_items, 
                         only_learn_from_buffer = instructions_file$only_learn_from_buffer, 
                         only_predict_from_buffer = instructions_file$only_predict_from_buffer, 
                         stm_weight = instructions_file$stm_weight, stm_duration = instructions_file$stm_duration, 
                         ltm_weight = instructions_file$ltm_weight, ltm_half_life = instructions_file$ltm_half_life, ltm_asymptote = instructions_file$ltm_asymptote, 
                         noise = instructions_file$noise, 
                         seed = instructions_file$seed
                         )
    results <- model_seq(mod, input_sequence, time=input_time_seq)
  } else if (model_type == "SIMPLE") {
    mod <- new_ppm_simple(alphabet_levels = alphabet_levels,
                          
                          order_bound = instructions_file$order_bound,
                          
                          shortest_deterministic = instructions_file$shortest_deterministic,
                          exclusion = instructions_file$exclusion,
                          update_exclusion = instructions_file$update_exclusion,
                          escape = instructions_file$escape
                          )
    results <- model_seq(mod, input_sequence)
  }

  
  # Convert $distribution to string of comma separated values
  results$distribution <- lapply(results$distribution, toString)
  results$distribution <- as.character(results$distribution)

  # Write results file data
  results_file_data_path <- paste(gsub("\\.feather", "", instructions_file$results_file_path), "-data.feather", sep="")
  write_feather(results, results_file_data_path)
  
  # Write results file
  meta_information = df <- data.frame(
    model_type=character(),
    alphabet_levels=character(),
    instructions_file_path=character(),
    results_file_data_path=character()
  )
  meta_information[1, ] = c(
    model_type,
    toString(alphabet_levels),
    instructions_file_path,
    results_file_data_path
  )
  write.csv(meta_information, instructions_file$results_file_path)
  
  # Return results_file_path
  return(instructions_file$results_file_path)
}

