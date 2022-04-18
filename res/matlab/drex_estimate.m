function out = drex_estimate(input_file_path)
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath, "./drex/"));

input = load(input_file_path);

drex_out = estimate_suffstat(input.xs, input.params);
distribution = input.params.distribution;

save(input.output_file_path, "drex_out", "input_file_path", "distribution");
out.output_file_path = input.output_file_path;
end