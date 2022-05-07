function out = drex_estimate(input_file_path)
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath, "../../models/DREX-model/"));

input = load(input_file_path);

drex_out = estimate_suffstat(input.xs, input.params);
distribution = input.params.distribution;
D = input.params.D;

save(input.output_file_path, "drex_out", "input_file_path", "distribution", "D");
out.output_file_path = input.output_file_path;
end