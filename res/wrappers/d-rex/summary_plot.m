function out = summary_plot(data_frame_file_path)
% load data
d = load(data_frame_file_path);
df = d.data_frame;

observations = df.observation;
ntime = length(observations);
observation_levels = unique(observations);

ppm_alphabet_size = df.ppm_alphabet_size;
ppm_predictions = reshape(cell2mat(df.ppm_predictions), ppm_alphabet_size, []);
ppm_information_content = df.ppm_information_content;
ppm_entropy = df.ppm_entropy;
ppm_model_order = df.ppm_model_order;

drex_predictions = reshape(cell2mat(df.drex_predictions), length(observation_levels), []);
drex_surprisal = df.drex_surprisal;
drex_entropy = df.drex_entropy;
drex_context_beliefs = reshape(cell2mat(df.drex_context_beliefs), length(observations)+1, []);
drex_context_beliefs = drex_context_beliefs';

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
ax(2) = subplot(subplotMaxN,1,2);
pos{2} = ax(2).Position;
[X, Y] = meshgrid(1:ntime, 1:ppm_alphabet_size);
contourf(X,Y,ppm_predictions,30,'fill','on','linestyle','-','linecolor','none');
caxis([0 1]);
set(gca,'colormap',cmap);
title("PPM: Predictions")
xlabel("Time \rightarrow")
ylabel("Observation [obs. index]")
colorbar()
ax(2).Position = pos{2};
%
ax(3) = subplot(subplotMaxN,1,3);
pos{3} = ax(3).Position;
[X, Y] = meshgrid(1:ntime, 1:length(observation_levels));
hold all;
contourf(X,Y,drex_predictions,30,'fill','on','linestyle','-','linecolor','none');
caxis([0 1]);
set(gca,'colormap',cmap);
hold off;
title("D-REX: Predictions")
xlabel("Time \rightarrow")
ylabel("Observation [obs. index]")
colorbar()
ax(3).Position = pos{3};

% subplot: PPM IC / DREX surprisal
ylim_value = max(max(ppm_information_content, [], 'all'), max(drex_surprisal, [], 'all'));
ylim_value = [0 round(ylim_value+0.1*ylim_value)];
subplot(subplotMaxN,1,4)
plot(ppm_information_content)
xlim([1 ntime]);
ylim(ylim_value);
title("PPM: Information Content")
xlabel("Time \rightarrow")
ylabel("Information Content [bits]")
%
subplot(subplotMaxN,1,5)
plot(drex_surprisal)
xlim([1 ntime]);
ylim(ylim_value);
title("D-REX: Surprisal")
xlabel("Time \rightarrow")
ylabel("Surprisal [bits]")

% subplots: PPM / DREX entropy
ylim_value = max(max(ppm_entropy, [], 'all'), max(drex_entropy, [], 'all'));
ylim_value = [0 round(ylim_value+0.1*ylim_value)];
subplot(subplotMaxN,1,6)
plot(ppm_entropy)
xlim([1 ntime]);
ylim(ylim_value);
title("PPM-Decay: Entropy")
xlabel("Time \rightarrow")
ylabel("Entropy [bits]")
%
subplot(subplotMaxN,1,7)
plot(drex_entropy)
xlim([1 ntime]);
ylim(ylim_value);
title("D-REX: Entropy")
xlabel("Time \rightarrow")
ylabel("Entropy [bits]")

% subplots: PPM / DREX interna
% subplot: PPM model order
%subplot(subplotMaxN,1,8)
%plot(ppm_model_order)
%title("PPM-Decay: Model Order")
%xlabel("Time \rightarrow")
%ylabel("Model Order")
%xlim([1 ntime]);
% subplot: DREX cb
ax(8) = subplot(subplotMaxN,1,8);
pos{8} = ax(8).Position;
drex_context_beliefs(drex_context_beliefs==0) = nan;

p = pcolor(drex_context_beliefs);
set(gca,'Color', ones(1,3),'colormap',parula(10),'ColorScale','log', 'CLim', [1e-4 1e0]);
p.LineStyle = 'none';
axis xy;
title('D-REX: Context Belief')
xlim([1 ntime]);
xlabel("Time \rightarrow")
ylabel("Context Windows")
set(gca,'YTick',xticks,'YTickLabel',xticklabels);
grid on;
colorbar()
ax(8).Position = pos{8};

% subplot: DREX bd
subplot(subplotMaxN,1,9)
post_bd = cell2mat(df.drex_bd);
plot(post_bd);
xlim([1 ntime]);
ylim([0 1]);
title('D-REX: Belief Dynamics')
xlabel("Time \rightarrow")
ylabel("Divergence [bits]")

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
ylabel("Change Probability")

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