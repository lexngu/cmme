# https://rpy2.github.io/doc/v3.4.x/html/robjects_rpackages.html#importing-arbitrary-r-code-as-a-package
library(ppm)
library(dplyr, warn.conflicts=FALSE)
library(tidyr)
run_ppm <- function(input_file_path) {
params <- read.csv(input_file_path)

params$only_learn_from_buffer <- as.logical(params$only_learn_from_buffer)
params$only_predict_from_buffer <- as.logical(params$only_predict_from_buffer)
params$debug_decay <- as.logical(params$debug_decay)
params$debug_smooth <- as.logical(params$debug_smooth)

alphabet_levels <- strsplit(gsub("\\[(.*?)\\]", "\\1", params$alphabet_levels), ",\\s*")[[1]]
if (length(alphabet_levels) > 0) {
  alphabet_levels <- as.factor(alphabet_levels)
} else {
  alphabet_levels <- character()
}
input_sequence <- strsplit(gsub("\\[(.*?)\\]", "\\1", params$input_sequence), ",\\s*")[[1]]
if (length(alphabet_levels) > 0) {
  input_sequence <- as.factor(input_sequence, levels = params$alphabet_levels)
} else {
  input_sequence <- as.factor(input_sequence)
}
input_time_seq <- strsplit(gsub("\\[(.*?)\\]", "\\1", params$input_time_sequence), ",\\s*")[[1]]
input_time_seq <- as.numeric(input_time_seq)

########################

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
                     debug_smooth = params$debug_smooth, 
                     debug_decay = params$debug_decay, 
                     alphabet_levels = alphabet_levels
                     )
res <- model_seq(mod, input_sequence, time=input_time_seq)
res <- res %>% mutate(observation = levels(input_sequence)[symbol])
#########################

ppm_res <- res %>% mutate(symbol_idx = row_number()) %>% 
  unnest_longer(distribution, indices_to = "distribution_idx") %>% 
  rowwise() %>% mutate(distribution_symbol = levels(input_sequence)[distribution_idx]) %>%
  select(symbol_idx, distribution_idx, distribution_symbol, distribution, everything())

write.csv(ppm_res, params$output_parameters_file_path)
params$output_csv_file
}
