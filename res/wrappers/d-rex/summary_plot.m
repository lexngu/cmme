function out = summary_plot(data_frame_file_path)
% load data
d = load(data_frame_file_path);
df = d.df;
[~,drex_file_name] = fileparts(d.drex_output);
[~,ppm_file_name] = fileparts(d.ppm_output);

x = cell2mat(df.observation_idx); x = x(1:end-1);
ntime = length(x)+1;
if iscell(df.observation)
    observations = cell2mat(df.observation);
else
    observations = df.observation;
end
observation_levels = unique(observations);

ppm_alphabet_size = df.ppm_alphabet_size{1};
ppm_predictions = reshape(cell2mat(df.ppm_probability_distribution), ppm_alphabet_size, []);
ppm_information_content = cell2mat(df.ppm_information_content);
ppm_entropy = cell2mat(df.ppm_entropy);
ppm_model_order = cell2mat(df.ppm_model_order);

drex_predictions = reshape(cell2mat(df.drex_predictions), length(observation_levels), []);
drex_surprisal = cell2mat(df.drex_surprisal);
drex_entropy = cell2mat(df.drex_entropy);
drex_context_beliefs = reshape(cell2mat(df.drex_context_beliefs), length(observations)+1, []);

% setup plot

cmap = zeros(256,3);
cmap(:,1) = linspace(1,0.4,256)';
cmap(:,2) = linspace(1,0.4,256)';
cmap(:,3) = 1;


% plot
f = figure;
f.PaperUnits = 'inches';
f.PaperSize = [4 2];
f.PaperPositionMode = 'manual';
f.PaperPosition = [0 0 15 30];

set(groot,'defaultLineLineWidth',2.0)

subplotMaxN = 10;
%annotation('textbox', [0, 0.2, 0, 0], 'string', append("PPM-Decay: ", ppm_file_name, " / D-REX: ", drex_file_name), 'Interpreter', 'none');
% subplot: Observation
subplot(subplotMaxN,1,1)
plot_sequence(observations);
title("Input Sequence")
xlabel("Time \rightarrow")
ylabel("Observation")

% subplots: PPM / DREX predictions
subplot(subplotMaxN,1,2)
[X, Y] = meshgrid(1:ntime, 1:ppm_alphabet_size);
contourf(X,Y,ppm_predictions,30,'fill','on','linestyle','-','linecolor','none');
set(gca,'colormap',cmap);
title("PPM: Predictions")
xlabel("Time \rightarrow")
ylabel("Observation")
%
subplot(subplotMaxN,1,3)
[X, Y] = meshgrid(1:ntime, 1:length(observation_levels));
hold all;
contourf(X,Y,drex_predictions,30,'fill','on','linestyle','-','linecolor','none');
set(gca,'colormap',cmap);
hold off;
title("D-REX: Predictions")
xlabel("Time \rightarrow")
ylabel("Observation")

% subplot: PPM IC / DREX surprisal
ylim_value = max(max(ppm_information_content, [], 'all'), max(drex_surprisal, [], 'all'));
ylim_value = [0 round(ylim_value+0.1*ylim_value)];
subplot(subplotMaxN,1,4)
plot(ppm_information_content)
xlim([1 ntime]);
ylim(ylim_value);
title("PPM: Information Content")
xlabel("Time \rightarrow")
ylabel("Information Content")
%
subplot(subplotMaxN,1,5)
plot(drex_surprisal)
xlim([1 ntime]);
ylim(ylim_value);
title("D-REX: Surprisal")
xlabel("Time \rightarrow")
ylabel("Surprisal")

% subplots: PPM / DREX entropy
ylim_value = max(max(ppm_entropy, [], 'all'), max(drex_entropy, [], 'all'));
%ylim_value = [0 round(ylim_value+0.1*ylim_value)];
subplot(subplotMaxN,1,6)
plot(ppm_entropy)
xlim([1 ntime]);
ylim([0 inf]);
title("PPM-Decay: Entropy")
xlabel("Time \rightarrow")
ylabel("Entropy")
%
subplot(subplotMaxN,1,7)
plot(drex_entropy)
xlim([1 ntime]);
ylim([0 inf]);
title("D-REX: Entropy")
xlabel("Time \rightarrow")
ylabel("Entropy")

% subplots: PPM / DREX interna
% subplot: PPM model order
%subplot(subplotMaxN,1,8)
%plot(ppm_model_order)
%title("PPM-Decay: Model Order")
%xlabel("Time \rightarrow")
%ylabel("Model Order")
%xlim([1 ntime]);
% subplot: DREX cb
subplot(subplotMaxN,1,8)
drex_context_beliefs(drex_context_beliefs==0) = nan;

p = pcolor(log10(drex_context_beliefs));
set(gca,'Color',0.95*ones(1,3),'colormap',parula(10));
p.LineStyle = 'none';
axis xy;
title('D-REX: Context Belief')
xlim([1 ntime]);
caxis([-5 0])
xlabel("Time \rightarrow")
ylabel("Context Belief")
set(gca,'YTick',xticks,'YTickLabel',xticklabels);
grid on;

% subplot: DREX bd
subplot(subplotMaxN,1,9)
post_bd = cell2mat(df.drex_bd);
plot(post_bd);
xlim([1 ntime]);
ylim([0 1]);
title('D-REX: Belief Dynamics')
xlabel("Time \rightarrow")

% subplot: DREX cd
subplot(subplotMaxN,1,10)
post_cd = cell2mat(df.drex_cd_probability);
changepoint = df.drex_cd_changepoint;
if iscell(changepoint)
    changepoint = cell2mat(df.drex_cd_changepoint);
end
changepoint = changepoint(1);
plot(post_cd);
xlim([1 ntime]);
ylim([0 1]);
hold all;
if ~isnan(changepoint)
    xline(changepoint, "-", ["Changepoint" num2str(changepoint)], 'LineWidth', 3);
end
hold off;
title('D-REX: Change Probability')
xlabel("Time \rightarrow")

out = "done";
end

% functions
% adapted version of "display_DREX_output.m"
function [xpos, ypos] = plot_sequence(sequence)
sequence = reshape(sequence,[],1);
sequence_length = length(sequence);

xpos = reshape([1:sequence_length; (1:sequence_length)+.15/.175; nan(1, sequence_length)],1,[]);
ypos = reshape([sequence sequence nan(sequence_length, 1)]',1,[]);

plot(xpos, ypos, 'LineWidth', 3);
xlim([1 sequence_length+1]);
end