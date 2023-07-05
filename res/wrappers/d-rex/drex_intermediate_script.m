function out = drex_intermediate_script(instructions_file_path)

if isfile(replace(instructions_file_path, "instructionsfile", "resultsfile"))
    fprintf("Results file at %s already exists. Abort.\n", instructions_file_path);
    return;
end

% load D-REX
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath, "../../models/DREX-model/"));

% read instructions file
if exist('OCTAVE_VERSION', 'builtin') == 0 %is MATLAB%
    instructions_file_path = convertStringsToChars(instructions_file_path);
end
instructions_file = load(instructions_file_path);

% set working directory
cd(fileparts(instructions_file_path));

out.results_file_path = instructions_file.results_file_path;
if isfile(instructions_file.results_file_path)
    fprintf("Results file at %s already exists. Abort.\n", instructions_file.results_file_path);
    cd(mfilepath);
    return;
end

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
    if strcmp(rdm_instructions.params.distribution, "gaussian") | strcmp(rdm_instructions.params.distribution, "gmm") | strcmp(rdm_instructions.params.distribution, "lognormal")
        for f = 1:nfeature
            if contains(instructions_file_path, "-cpitch-")
                positions = linspace(0, 127, 128);
            elseif contains(instructions_file_path, "-freq-")
                positions = [8.18, 8.66, 9.18, 9.72, 10.3, 10.91, 11.56, 12.25, 12.98, 13.75, 14.57, 15.43, 16.35, 17.32, 18.35, 19.45, 20.6, 21.83, 23.12, 24.5, 25.96, 27.5, 29.14, 30.87, 32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49.0, 51.91, 55.0, 58.27, 61.74, 65.41, 69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98.0, 103.83, 110.0, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.0, 196.0, 207.65, 220.0, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.3, 440.0, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61, 880.0, 932.33, 987.77, 1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.0, 1864.66, 1975.53, 2093.0, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.0, 3729.31, 3951.07, 4186.01, 4434.92, 4698.64, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040.0, 7458.62, 7902.13, 8372.02, 8869.84, 9397.27, 9956.06, 10548.08, 11175.3, 11839.82, 12543.85];
            else 
                positions = reshape(unique(input_sequence(:,f)), 1, []);
            end
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
    save(instructions_file.results_file_path, "-v7", "instructions_file_path", "input_sequence", "distribution", "estimate_suffstat_results", "run_DREX_model_results", "post_DREX_changedecision_results", "post_DREX_prediction_results", "post_DREX_beliefdynamics_results", "post_DREX_changedecision_results", "change_decision_threshold");
end

cd(mfilepath);
end