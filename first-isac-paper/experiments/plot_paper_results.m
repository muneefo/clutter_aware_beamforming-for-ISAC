function stats = plot_paper_results(folder)
% Generate paired statistics and two PDF figures per condition from saved CSVs.
% Example: stats = plot_paper_results(fullfile(repo, 'paper_mc_v1'));
if nargin == 0
    folder = uigetdir(pwd, 'Select the folder containing scenes.csv and learning.csv');
    if isequal(folder, 0)
        stats = table();
        return
    end
end

scenesFile = fullfile(folder, 'scenes.csv');
learningFile = fullfile(folder, 'learning.csv');
assert(isfile(scenesFile), 'Missing file: %s', scenesFile);
assert(isfile(learningFile), 'Missing file: %s', learningFile);

scenes = readtable(scenesFile, 'TextType', 'string');
learning = readtable(learningFile, 'TextType', 'string');
scenes.aoValid = readFlag(scenes.aoValid);
scenes.hit = readFlag(scenes.hit);
learning.learnedOnlyHit = readFlag(learning.learnedOnlyHit);
learning.fixedOnlyHit = readFlag(learning.fixedOnlyHit);

conditionNames = {'K', 'gammaDB'};
budgetNames = {'outer', 'inner'};
groupNames = [conditionNames, budgetNames];
stats = unique(learning(:, groupNames), 'rows');
groups = height(stats);
stats.pairedN = zeros(groups, 1);
stats.medianGainDB = nan(groups, 1);
stats.bootstrapLow = nan(groups, 1);
stats.bootstrapHigh = nan(groups, 1);
stats.learnedOnlyHits = zeros(groups, 1);
stats.fixedOnlyHits = zeros(groups, 1);

stream = RandStream('mt19937ar', 'Seed', 7331);
for i = 1:groups
    sameCondition = learning.K == stats.K(i) & learning.gammaDB == stats.gammaDB(i);
    sameBudget = learning.outer == stats.outer(i) & learning.inner == stats.inner(i);
    rows = learning(sameCondition & sameBudget, :);
    gain = rows.learnedMinusFixedDB;
    gain = gain(isfinite(gain));
    n = numel(gain);
    stats.pairedN(i) = n;
    stats.learnedOnlyHits(i) = nnz(rows.learnedOnlyHit);
    stats.fixedOnlyHits(i) = nnz(rows.fixedOnlyHit);
    if n > 0
        stats.medianGainDB(i) = median(gain);
    end
    if n > 1
        indices = randi(stream, n, n, 2000);
        draws = reshape(gain(indices), n, 2000);
        medians = sort(median(draws, 1));
        stats.bootstrapLow(i) = medians(50);
        stats.bootstrapHigh(i) = medians(1950);
    end
end
writetable(stats, fullfile(folder, 'paired_intervals.csv'));

conditions = unique(scenes(:, conditionNames), 'rows');
for i = 1:height(conditions)
    users = conditions.K(i);
    threshold = conditions.gammaDB(i);
    inScenes = scenes.K == users & scenes.gammaDB == threshold;
    inLearning = learning.K == users & learning.gammaDB == threshold;
    suffix = sprintf('K%d_G%g.pdf', users, threshold);
    plotQuality(scenes(inScenes, :), fullfile(folder, ['quality_time_' suffix]));
    plotGain(learning(inLearning, :), fullfile(folder, ['learning_gain_' suffix]));
end
disp(stats)
end

function flags = readFlag(values)
% CSVs can store a logical column as 0/1 or as true/false.
words = lower(strtrim(string(values)));
valid = ismember(words, ["0", "1", "false", "true"]);
assert(all(valid), 'Expected a 0/1 or true/false column in the saved CSV.');
flags = ismember(words, ["1", "true"]);
end

function plotQuality(scenes, filename)
fig = figure('Color', 'white', 'Visible', 'off', 'Position', [100 100 900 500]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
ax = axes('Parent', fig);
hold(ax, 'on');
modes = ["physics", "fixedProposal", "learnedProposal"];
styles = {'o-', 's--', '^:'};
for i = 1:numel(modes)
    rows = scenes(scenes.mode == modes(i), :);
    budgets = unique(rows(:, {'outer', 'inner'}), 'rows');
    hitRate = nan(height(budgets), 1);
    seconds = nan(height(budgets), 1);
    for b = 1:height(budgets)
        sameBudget = rows.outer == budgets.outer(b) & rows.inner == budgets.inner(b);
        comparable = sameBudget & rows.aoValid;
        if any(comparable)
            hitRate(b) = mean(rows.hit(comparable));
            seconds(b) = median(rows.seconds(comparable));
        end
    end
    plot(ax, seconds, hitRate, styles{i}, 'LineWidth', 1.3, 'DisplayName', char(modes(i)));
end
xlabel(ax, 'Median design time (s)');
ylabel(ax, 'Fraction meeting the AO quality target');
ylim(ax, [-0.04 1.04]);
legend(ax, 'Location', 'best');
grid(ax, 'on');
exportgraphics(fig, filename, 'ContentType', 'vector');
end

function plotGain(rows, filename)
fig = figure('Color', 'white', 'Visible', 'off', 'Position', [100 100 900 500]);
cleanup = onCleanup(@() close(fig)); %#ok<NASGU>
ax = axes('Parent', fig);
hold(ax, 'on');
budgets = unique(rows(:, {'outer', 'inner'}), 'rows');
hasPairs = false;
for b = 1:height(budgets)
    sameBudget = rows.outer == budgets.outer(b) & rows.inner == budgets.inner(b);
    gain = rows.learnedMinusFixedDB(sameBudget);
    gain = sort(gain(isfinite(gain)));
    n = numel(gain);
    if n == 0
        continue
    end
    probability = (1:n).' / n;
    name = sprintf('%d outer x %d inner, n=%d', budgets.outer(b), budgets.inner(b), n);
    stairs(ax, [gain(1); gain], [0; probability], 'LineWidth', 1.3, 'DisplayName', name);
    hasPairs = true;
end
xline(ax, 0, ':', 'HandleVisibility', 'off');
xlabel(ax, 'True SCNR, learned minus fixed proposal (dB)');
ylabel(ax, 'Cumulative fraction of paired feasible channels');
ylim(ax, [0 1.02]);
grid(ax, 'on');
if hasPairs
    legend(ax, 'Location', 'best');
end
exportgraphics(fig, filename, 'ContentType', 'vector');
end
