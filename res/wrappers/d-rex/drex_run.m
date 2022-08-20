function out = drex_run(instructions_file_path)
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath, "../../models/DREX-model/"));

instructions_file_path = convertStringsToChars(instructions_file_path);
input = load(instructions_file_path);
input_sequence = input.x;
params = input.params;
model_as_string = input.model_as_string

% matlab.engine cannot create n x 1-cells, only 1 x n-cells. This affects the prior, and needs to be fixed manually.
prior_field_names = fieldnames(params.prior);
for k=1:numel(prior_field_names)
    field_value = params.prior.(prior_field_names{k});
    if iscell(field_value)
        cell_size = size(field_value);
        if cell_size(1,2) > 1
            params.prior.(prior_field_names{k}) = field_value';
        end
    end
end

drex_out = run_DREX_model(input_sequence, params);

drex_psi = [];
if params.distribution == "gaussian" | params.distribution == "gmm" | params.distribution == "lognormal"
    f = 1;
    pred_pos = reshape(unique(input_sequence), 1, []);
    drex_psi = post_DREX_prediction(f, drex_out, pred_pos);
end

drex_cd_threshold = 0.09; %TODO remove this hard-coded variable
drex_cd = post_DREX_changedecision(drex_out, drex_cd_threshold); % ToDo expose threshold to input file.
drex_bd = post_DREX_beliefdynamics(drex_out);

drex_out.context_beliefs = drex_out.context_beliefs';
drex_psi = drex_psi';

save(input.results_file_path, "instructions_file_path", "model_as_string", "input_sequence", "drex_out", "drex_psi", "drex_cd", "drex_cd_threshold", "drex_bd");
out.results_file_path = input.results_file_path;
end