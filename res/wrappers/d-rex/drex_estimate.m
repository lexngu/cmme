function out = drex_estimate(instructions_file_path)
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath, "../../models/DREX-model/"));

input = load(instructions_file_path);

drex_out = estimate_suffstat(input.xs, input.params);
distribution = input.params.distribution;
D = input.params.D;

save(input.results_file_path, "drex_out", "instructions_file_path", "distribution", "D");
out.results_file_path = input.results_file_path;
end