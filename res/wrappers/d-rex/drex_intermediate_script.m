function out = drex_intermediate_script(instructions_file_path)
% load D-REX
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath, "../../models/DREX-model/"));

% read instructions file
instructions_file_path = convertStringsToChars(instructions_file_path);
instructions_file = load(instructions_file_path);

% calculate prior (if requested)
estimate_suffstat_results = struct;
if isfield(instructions_file, "estimate_suffstat")
    es_instructions = instructions_file.estimate_suffstat;
    distribution = es_instructions.params.distribution;
    estimate_suffstat_results = estimate_suffstat(es_instructions.xs, es_instructions.params);

    if ~isfield(instructions_file, "run_DREX_model")
        save(instructions_file.results_file_path, "instructions_file_path", "estimate_suffstat_results", "distribution");
    end
end

% run DREX (if requested)
if isfield(instructions_file, "run_DREX_model")
    rdm_instructions = instructions_file.run_DREX_model;
    input_sequence = rdm_instructions.x;
    
    [~, nfeature] = size(input_sequence);
    
    if isfield(estimate_suffstat_results, "n")
        % plug-in calculated prior into params for run_DREX_model.m
        rdm_instructions.params.prior = estimate_suffstat_results;
    else
        % matlab.engine cannot create n x 1-cells, only 1 x n-cells. This affects the prior, and needs to be fixed manually.
        prior_field_names = fieldnames(rdm_instructions.params.prior);
        for k=1:numel(prior_field_names)
            field_value = rdm_instructions.params.prior.(prior_field_names{k});
            if iscell(field_value)
                cell_size = size(field_value);
                if cell_size(1,2) > 1
                    rdm_instructions.params.prior.(prior_field_names{k}) = field_value';
                end
            end
        end
    end

    % invoke D-REX (run_DREX_model.m)
    run_DREX_model_results = run_DREX_model(input_sequence, rdm_instructions.params);
    
    % calculate marginal (predictive) prob. distribution (post_DREX_prediction.m)
    post_DREX_prediction_results = cell(nfeature,1);
    if rdm_instructions.params.distribution == "gaussian" | rdm_instructions.params.distribution == "gmm" | rdm_instructions.params.distribution == "lognormal"
        for f = 1:nfeature
            positions = reshape(unique(input_sequence(:,f)), 1, []);
            
            post_DREX_prediction_results{f}.positions = positions;
            post_DREX_prediction_results{f}.prediction = post_DREX_prediction(f, run_DREX_model_results, positions)';
        end
    end
    
    % calculate belief dynamics (post_DREX_beliefdynamics.m)
    post_DREX_beliefdynamics_results = post_DREX_beliefdynamics(run_DREX_model_results);
    
    % calculate changedecision (post_DREX_changedecision.m)
    pdc_instructions = instructions_file.post_DREX_changedecision;
    change_decision_threshold = pdc_instructions.threshold;
    post_DREX_changedecision_results = post_DREX_changedecision(run_DREX_model_results, change_decision_threshold);

    distribution = rdm_instructions.params.distribution;
    save(instructions_file.results_file_path, "instructions_file_path", "input_sequence", "distribution", "estimate_suffstat_results", "run_DREX_model_results", "post_DREX_changedecision_results", "post_DREX_prediction_results", "post_DREX_beliefdynamics_results", "post_DREX_changedecision_results", "change_decision_threshold");
end

out.results_file_path = instructions_file.results_file_path;
end