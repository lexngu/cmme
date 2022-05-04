function out = drex_run(input_file_path)
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath, "../../models/DREX-model/"));

input = load(input_file_path);
input_sequence = input.x;
params = input.params;

if isfield(params,'D')
    params.D = double(params.D);
end

if params.distribution == "gmm"
    % convert to cell arrays; needed from run_DREX_model (case: GMM)
    prior_field_names = fieldnames(params.prior);
    for k=1:numel(prior_field_names)
        field_value = params.prior.(prior_field_names{k});
        if ~iscell(field_value)
            params.prior.(prior_field_names{k}) = num2cell(field_value, 2);
        end
    end
elseif params.distribution == "gaussian" % some conversion, since i cannot output structs from python.
    prior_mu{1} = double(cell2mat(params.prior.mu))';
    prior_ss{1} = reshape(double(cell2mat(params.prior.ss)), [], size(params.prior.ss, 2));
    prior_n{1} = double(cell2mat(params.prior.n));

    params.prior.mu = prior_mu;
    params.prior.ss = prior_ss;
    params.prior.n = prior_n;
end


drex_out = run_DREX_model(input_sequence, params);

f = 1; 
pred_pos = reshape(unique(input_sequence), 1, []);
drex_psi = post_DREX_prediction(f, drex_out, pred_pos);
drex_cd_threshold = 0.09;
drex_cd = post_DREX_changedecision(drex_out, drex_cd_threshold); % ToDo expose threshold to input file.
drex_bd = post_DREX_beliefdynamics(drex_out);

drex_out.context_beliefs = drex_out.context_beliefs';
drex_psi = drex_psi';

save(input.output_file_path, "input_file_path", "input_sequence", "drex_out", "drex_psi", "drex_cd", "drex_cd_threshold", "drex_bd");
out.output_file_path = input.output_file_path;
end