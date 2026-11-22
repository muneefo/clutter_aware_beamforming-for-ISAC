function figures = plot_full_isac_results(folder)
% Display three figures and export a vector PDF and PNG of each.
% plot_full_isac_results(R.folder) uses the original MATLAB CSV precision.
% With no argument, use the two CSVs packaged alongside this file.
if nargin == 0
    folder = fileparts(mfilename('fullpath'));
end
S = readtable(fullfile(folder, 'summary.csv'), 'TextType', 'string');
I = readtable(fullfile(folder, 'paired_intervals.csv'), 'TextType', 'string');
conditions = unique(S(:, {'K', 'gammaDB'}), 'rows');
budgets = unique(S.outer(S.mode ~= "AO") .* S.inner(S.mode ~= "AO"));
S.budget = S.outer .* S.inner;
I.budget = I.outer .* I.inner;
out = fullfile(folder, 'publication_figures');
if ~isfolder(out)
    mkdir(out);
end
palette = [0.15 0.30 0.44; 0.68 0.31 0.14; 0.09 0.49 0.44];
modes = ["physics", "fixedProposal", "learnedProposal"];
names = {'Physics only', 'Fixed proposal', 'Learned proposal'};
markers = {'o', 's', '^'};
n = height(conditions);
x = 1:numel(budgets);
figures = gobjects(3,1);

% Same hardware and channels for all methods. Each group shows the median time.
figures(1) = figure('Name', 'Design runtime', 'Color', 'w', 'Position', [80 80 1120 430]);
layout = tiledlayout(figures(1), 1, n, 'TileSpacing', 'compact');
for c = 1:n
    ax = nexttile(layout, c);
    hold(ax, 'on');
    rows = S(S.K == conditions.K(c) & S.gammaDB == conditions.gammaDB(c), :);
    h = plotModes(ax, rows, 'medianSeconds', budgets, modes, palette, markers);
    ao = rows(rows.mode == "AO", :);
    assert(height(ao) == 1, 'Expected one AO summary row per condition.');
    h(4) = plot(ax, x(end)+1, ao.medianSeconds, 'kd', 'MarkerFaceColor', 'k');
    labels = [string(budgets(:)); "AO"];
    set(ax, 'YScale', 'log', 'XTick', [x x(end)+1], 'XTickLabel', labels);
    validTimes = S.medianSeconds(isfinite(S.medianSeconds) & S.medianSeconds > 0);
    if ~isempty(validTimes)
        ylim(ax, [0.75*min(validTimes), 1.4*max(validTimes)]);
    end
    xlim(ax, [0.5 x(end)+1.5]); grid(ax, 'on');
    title(ax, sprintf('%d users, %g dB', conditions.K(c), conditions.gammaDB(c)));
    xlabel(ax, 'LP budget or AO reference');
    if c == 1
        ylabel(ax, 'Median design time (s)');
        lg = legend(ax, h, [names {'AO reference'}], 'Orientation', 'horizontal');
        lg.Layout.Tile = 'north';
    end
end
saveFigure(figures(1), out, 'runtime');

% Top: sensing gap for comparable references. Bottom: individual quality hits.
figures(2) = figure('Name', 'Sensing quality', 'Color', 'w', 'Position', [95 95 1120 660]);
layout = tiledlayout(figures(2), 2, n, 'TileSpacing', 'compact');
allGaps = S.medianGapDB(S.mode ~= "AO" & isfinite(S.medianGapDB));
for c = 1:n
    rows = S(S.K == conditions.K(c) & S.gammaDB == conditions.gammaDB(c), :);
    ax = nexttile(layout, c);
    hold(ax, 'on');
    h = plotModes(ax, rows, 'medianGapDB', budgets, modes, palette, markers);
    h(4) = yline(ax, 0.1, 'k--');
    if ~isempty(allGaps) && all(allGaps > 0)
        set(ax, 'YScale', 'log');
        ylim(ax, [0.7*min([0.1; allGaps]), 1.4*max([0.1; allGaps])]);
    elseif ~isempty(allGaps)
        low = min([0; allGaps]); high = max([0.1; allGaps]);
        ylim(ax, [low-0.05*(high-low), high+0.05*(high-low)]);
    end
    xlim(ax, [0.5 x(end)+0.5]);
    set(ax, 'XTick', x, 'XTickLabel', string(budgets)); grid(ax, 'on');
    title(ax, sprintf('%d users, %g dB', conditions.K(c), conditions.gammaDB(c)));
    xlabel(ax, 'LP budget');
    if c == 1
        ylabel(ax, 'Median AO gap (dB)');
        lg = legend(ax, h, [names {'0.1 dB target'}], 'Orientation', 'horizontal');
        lg.Layout.Tile = 'north';
    end
    ax = nexttile(layout, n+c);
    counts = nan(numel(modes), numel(budgets));
    valid = counts;
    for m = 1:numel(modes)
        r = sortrows(rows(rows.mode == modes(m), :), 'budget');
        counts(m,:) = r.hits.';
        valid(m,:) = r.validAO.';
    end
    fraction = counts ./ valid;
    imagesc(ax, fraction, [0 1]);
    colormap(ax, interp1([0 0.5 1], [0.96 0.97 0.98; 0.23 0.45 0.54; 0.04 0.23 0.32], linspace(0,1,128)));
    for m = 1:numel(modes)
        for b = 1:numel(budgets)
            ink = 'k';
            if fraction(m,b) >= 0.43
                ink = 'w';
            end
            label = sprintf('%d/%d', counts(m,b), valid(m,b));
            text(ax, b, m, label, 'HorizontalAlignment', 'center', 'Color', ink);
        end
    end
    set(ax, 'XTick', x, 'XTickLabel', string(budgets), 'YTick', 1:3);
    if c == 1
        ax.YTickLabel = names;
    else
        ax.YTickLabel = {};
    end
    xlabel(ax, 'LP budget');
end
title(layout, 'Sensing gap and quality hits per feasible AO reference');
saveFigure(figures(2), out, 'sensing_quality');

% Paired changes and bootstrap intervals are computed on feasible method pairs.
figures(3) = figure('Name', 'Paired learning effect', 'Color', 'w', 'Position', [110 110 820 470]);
ax = axes('Parent', figures(3)); hold(ax, 'on');
positions = numel(budgets):-1:1;
h = gobjects(n,1); labels = cell(n,1);
offset = ((n+1)/2 - (1:n)) * 0.19;
for c = 1:n
    r = I(I.K == conditions.K(c) & I.gammaDB == conditions.gammaDB(c), :);
    r = sortrows(r, 'budget');
    assert(isequal(r.budget(:), budgets(:)), 'Missing paired interval for one budget.');
    j = mod(c-1, size(palette,1))+1;
    med = 1000*r.medianGainDB;
    lo = 1000*r.bootstrapLow; hi = 1000*r.bootstrapHigh;
    h(c) = errorbar(ax, med, positions.'+offset(c), med-lo, hi-med, 'horizontal', ...
        'LineStyle', 'none', 'Marker', markers{j}, 'Color', palette(j,:), 'LineWidth', 1.2);
    labels{c} = sprintf('%d users, %g dB', conditions.K(c), conditions.gammaDB(c));
end
xline(ax, 0, 'k:', 'HandleVisibility', 'off');
set(ax, 'YTick', sort(positions), 'YTickLabel', string(flipud(budgets(:))));
ylim(ax, [0.5 numel(budgets)+0.5]);
xlabel(ax, 'Paired median true SCNR, learned minus fixed (10^{-3} dB)');
ylabel(ax, 'LP budget'); grid(ax, 'on');
legend(ax, h, labels, 'Location', 'northoutside', 'Orientation', 'horizontal');
saveFigure(figures(3), out, 'learning_effect');
fprintf('Figures saved in %s\n', out);
end

function h = plotModes(ax, rows, field, budgets, modes, palette, markers)
h = gobjects(1,numel(modes));
for m = 1:numel(modes)
    r = sortrows(rows(rows.mode == modes(m), :), 'budget');
    assert(isequal(r.budget(:), budgets(:)), 'Missing or repeated method/budget row.');
    x = (1:numel(budgets)) + (m-2)*0.2;
    face = palette(m,:);
    if m == 2
        face = 'white';
    end
    h(m) = plot(ax, x, r.(field), 'LineStyle', 'none', 'Marker', markers{m}, ...
        'MarkerSize', 6, 'Color', palette(m,:), 'MarkerFaceColor', face);
end
end

function saveFigure(fig, folder, name)
fig.Visible = 'on';
exportgraphics(fig, fullfile(folder, [name '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(folder, [name '.png']), 'Resolution', 220);
end
