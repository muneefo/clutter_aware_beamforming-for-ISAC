function summary = plot_matlab_sinr_sweep(runFolder,destination)
%PLOT_MATLAB_SINR_SWEEP Plot SCNR and full design time from a measured raw CSV.
% Each point uses the same eligible scenes across all compared methods.
arguments
    runFolder (1,1) string
    destination (1,1) string = ""
end
if destination == "", destination = fullfile(runFolder,'plots'); end
assert(~isfolder(destination) && ~isfile(destination),'Use a fresh plot destination.');
manifest = jsondecode(fileread(fullfile(runFolder,'environment.json')));
raw = readtable(fullfile(runFolder,'raw.csv'),'TextType','string');
assert(height(raw) > 0,'No timed records are available.');
assert(height(unique(raw(:,{'gammaDB','seed','rep','method'}))) == height(raw), ...
    'Duplicate timed trial key in raw.csv.');
if isnumeric(raw.feasible) || islogical(raw.feasible)
    raw.feasible = raw.feasible == 1;
else
    raw.feasible = ismember(lower(string(raw.feasible)),["1","true"]);
end
methods = reshape(string(manifest.methods),1,[]);
opts = manifest.options;
seeds = opts.Seeds(:);
gammas = sort(opts.SINRdB(:));
stream = RandStream('mt19937ar','Seed',manifest.bootstrapSeed);
nm = numel(methods);
ns = numel(seeds);
analytic = find(methods == "analytic",1);
ao = find(methods == "ao",1);
reference = analytic;
if isempty(reference), reference = find(methods == "reducedCost",1); end
assert(~isempty(reference),'Expected an analytic or reduced-cost comparison arm.');
rows = repmat(stats_row(),numel(gammas)*nm,1);
pairs = cell(numel(gammas)*nm,1);
for g = 1:numel(gammas)
    sub = raw(raw.gammaDB == gammas(g),:);
    values = nan(ns,nm,4);
    for m = 1:nm
        branch = sub(sub.method == methods(m),:);
        for i = 1:ns
            t = branch(branch.seed == seeds(i),:);
            complete = isequal(sort(t.rep(:)),(1:opts.Repeats).');
            finite = all(isfinite([t.designSeconds,t.designSCNRdB,t.trueSCNRdB]),'all');
            if complete && finite && all(t.feasible) && all(t.designSeconds > 0)
                perScene = median([t.designSCNRdB,t.trueSCNRdB,t.designSeconds,t.inferenceSeconds],1);
                values(i,m,:) = reshape(perScene,1,1,[]);
            end
        end
    end
    common = all(isfinite(values(:,:,3)),2);
    n = nnz(common);
    if n > 0, index = randi(stream,n,n,manifest.bootstrapDraws); end
    for m = 1:nm
        k = (g-1)*nm+m;
        row = stats_row();
        row.gammaDB = gammas(g);
        row.method = methods(m);
        row.referenceMethod = methods(reference);
        row.attemptedScenes = ns;
        row.completeFeasibleScenes = nnz(isfinite(values(:,m,3)));
        row.commonScenes = n;
        if n > 0
            score = values(common,m,1);
            trueScore = values(common,m,2);
            elapsed = values(common,m,3);
            inference = values(common,m,4);
            referenceTime = values(common,reference,3);
            designChange = score-values(common,reference,1);
            trueChange = trueScore-values(common,reference,2);
            timeRatio = elapsed./referenceTime;
            design = 10.^(score/10);
            ideal = 10.^(trueScore/10);
            row.designSCNRdB = 10*log10(mean(design));
            row.trueSCNRdB = 10*log10(mean(ideal));
            bounds = percentiles(10*log10(mean(resample(design,index),1)),[.025 .975]);
            row.designLow = bounds(1); row.designHigh = bounds(2);
            bounds = percentiles(10*log10(mean(resample(ideal,index),1)),[.025 .975]);
            row.trueLow = bounds(1); row.trueHigh = bounds(2);
            row.medianSeconds = median(elapsed);
            bounds = percentiles(median(resample(elapsed,index),1),[.025 .975]);
            row.timeLow = bounds(1); row.timeHigh = bounds(2);
            bounds = percentiles(elapsed,[.25 .75]);
            row.timeQ1 = bounds(1); row.timeQ3 = bounds(2);
            row.medianInferenceSeconds = median(inference);
            row.medianDesignDeltaDB = median(designChange);
            bounds = percentiles(median(resample(designChange,index),1),[.025 .975]);
            row.designDeltaLow = bounds(1); row.designDeltaHigh = bounds(2);
            row.medianTrueDeltaDB = median(trueChange);
            bounds = percentiles(median(resample(trueChange,index),1),[.025 .975]);
            row.trueDeltaLow = bounds(1); row.trueDeltaHigh = bounds(2);
            row.medianTimeRatio = median(timeRatio);
            bounds = percentiles(median(resample(timeRatio,index),1),[.025 .975]);
            row.timeRatioLow = bounds(1); row.timeRatioHigh = bounds(2);
            analyticTimeRatio = nan(n,1);
            if ~isempty(analytic)
                analyticTimeRatio = timeRatio;
                row.medianDesignDeltaVersusAnalyticDB = row.medianDesignDeltaDB;
                row.medianTrueDeltaVersusAnalyticDB = row.medianTrueDeltaDB;
                row.medianTimeRatioVersusAnalytic = row.medianTimeRatio;
                row.analyticTimeRatioLow = row.timeRatioLow;
                row.analyticTimeRatioHigh = row.timeRatioHigh;
            end
            aoSpeedup = nan(n,1);
            if ~isempty(ao)
                aoSpeedup = values(common,ao,3)./elapsed;
                row.medianAOSpeedup = median(aoSpeedup);
                bounds = percentiles(median(resample(aoSpeedup,index),1),[.025 .975]);
                row.speedupLow = bounds(1); row.speedupHigh = bounds(2);
            end
            pairs{k} = table(repmat(gammas(g),n,1),seeds(common),repmat(methods(m),n,1), ...
                designChange,trueChange,analyticTimeRatio,aoSpeedup, ...
                repmat(methods(reference),n,1),timeRatio,score,trueScore,elapsed,inference, ...
                'VariableNames',{'gammaDB','seed','method','designDeltaDB','trueDeltaDB', ...
                'timeRatioVersusAnalytic','speedupVersusAO','referenceMethod','timeRatio', ...
                'designSCNRdB','trueSCNRdB','designSeconds','inferenceSeconds'});
        end
        rows(k) = row;
    end
end
summary = struct2table(rows);
mkdir(destination);
writetable(summary,fullfile(destination,'summary.csv'));
available = ~cellfun(@isempty,pairs);
if any(available)
    writetable(vertcat(pairs{available}),fullfile(destination,'paired.csv'));
end
make_figures(summary,methods,destination);
end

function row = stats_row()
row = struct('gammaDB',NaN,'method',"",'attemptedScenes',0,'completeFeasibleScenes',0,'commonScenes',0, ...
    'designSCNRdB',NaN,'designLow',NaN,'designHigh',NaN,'trueSCNRdB',NaN,'trueLow',NaN,'trueHigh',NaN, ...
    'medianSeconds',NaN,'timeLow',NaN,'timeHigh',NaN,'timeQ1',NaN,'timeQ3',NaN, ...
    'medianInferenceSeconds',NaN,'medianAOSpeedup',NaN,'speedupLow',NaN,'speedupHigh',NaN, ...
    'medianDesignDeltaVersusAnalyticDB',NaN,'designDeltaLow',NaN,'designDeltaHigh',NaN, ...
    'medianTrueDeltaVersusAnalyticDB',NaN,'trueDeltaLow',NaN,'trueDeltaHigh',NaN, ...
    'medianTimeRatioVersusAnalytic',NaN,'analyticTimeRatioLow',NaN,'analyticTimeRatioHigh',NaN, ...
    'referenceMethod',"",'medianDesignDeltaDB',NaN,'medianTrueDeltaDB',NaN, ...
    'medianTimeRatio',NaN,'timeRatioLow',NaN,'timeRatioHigh',NaN);
end

function draws = resample(x,index)
draws = reshape(x(index),size(index));
end

function q = percentiles(x,p)
% Linear interpolation between order statistics, including one-scene pilots.
x = sort(x(:));
index = 1+(numel(x)-1)*p;
lo = floor(index);
hi = ceil(index);
lower = reshape(x(lo),size(p));
upper = reshape(x(hi),size(p));
q = lower+(index-lo).*(upper-lower);
end

function make_figures(S,methods,destination)
labels = struct('ao','AO–SDP','analytic','Analytic pricing','reducedCost','L2O reduced cost', ...
    'envelope','L2O envelope');
styles = struct('ao','--o','analytic','-s','reducedCost','-^','envelope','-d');
for kind = ["scnr","timing"]
    fig = figure('Visible','off','Color','w','Position',[100 100 980 392]);
    closeFigure = onCleanup(@() close(fig));
    layout = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
    left = nexttile(layout); hold(left,'on'); grid(left,'on');
    right = nexttile(layout); hold(right,'on'); grid(right,'on');
    for method = methods
        rows = S(S.method == method,:);
        key = char(method);
        style = styles.(key);
        label = labels.(key);
        if kind == "scnr"
            curve(left,rows.gammaDB,rows.designSCNRdB,rows.designLow,rows.designHigh,style,label);
            curve(right,rows.gammaDB,rows.trueSCNRdB,rows.trueLow,rows.trueHigh,style,label);
        else
            curve(left,rows.gammaDB,1e3*rows.medianSeconds,1e3*rows.timeLow,1e3*rows.timeHigh,style,label);
            plot(right,rows.gammaDB,100*rows.completeFeasibleScenes./rows.attemptedScenes, ...
                style,'DisplayName',label,'LineWidth',1.35,'MarkerSize',4);
        end
    end
    if kind == "scnr"
        ylabel(left,'SCNR under design kernel (dB)');
        ylabel(right,'SCNR under ideal kernel (dB)');
        title(layout,'Common feasible scenes · linear-domain means and 95% intervals');
        filename = 'scnr-vs-sinr.pdf';
    else
        ylabel(left,'Median design time (ms)');
        set(left,'YScale','log');
        ylabel(right,'Complete feasible scenes (%)');
        ylim(right,[0 104]);
        title(layout,'Design time on common scenes · coverage over all requested seeds');
        filename = 'timing-vs-sinr.pdf';
    end
    xlabel(left,'Required SINR (dB)'); xlabel(right,'Required SINR (dB)');
    legendBox = legend(left,'Orientation','horizontal');
    legendBox.Layout.Tile = 'south';
    exportgraphics(fig,fullfile(destination,filename),'ContentType','vector');
    clear closeFigure
end
end

function curve(ax,x,y,lo,hi,style,label)
errorbar(ax,x,y,max(0,y-lo),max(0,hi-y),style, ...
    'DisplayName',label,'LineWidth',1.35,'MarkerSize',4,'CapSize',5);
end
