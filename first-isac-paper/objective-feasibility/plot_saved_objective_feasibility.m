function F = plot_saved_objective_feasibility(folder)
% Display absolute SCNR, physical feasibility and constraint margins from raw.csv.
% F = plot_saved_objective_feasibility(R.folder)
% Also accepts R.raw, or a results struct R containing .raw and .folder.
if nargin == 0
    folder = uigetdir(pwd,'Select the paper run folder containing raw.csv');
    if isequal(folder,0), F = gobjects(0); return; end
end
if istable(folder)
    T = folder; folder = pwd;
elseif isstruct(folder)
    T = folder.raw; folder = folder.folder;
else
    T = readtable(fullfile(folder,'raw.csv'),'TextType','string');
end
keys = {'K','gammaDB','scene','mode','outer','inner'};
fields = [keys,{'rep','trueDB','designDB','marginDB','powerW','feasible'}];
assert(all(ismember(fields,T.Properties.VariableNames)),'Select raw.csv from run_paper_experiments.');
assert(height(unique(T(:,[keys,{'rep'}])))==height(T),'Duplicate timing-repeat records.');
T.mode = string(T.mode);
flags = lower(string(T.feasible));
assert(all(ismember(flags,["true","false","1","0"])),'Unrecognized feasible flag.');
groupFeasible = ismember(flags,["true","1"]);
[G,S] = findgroups(T(:,keys));
S.feasible = splitapply(@all,groupFeasible,G);
S.designDB = splitapply(@median,T.designDB,G);
S.trueDB = splitapply(@median,T.trueDB,G);
S.marginDB = splitapply(@(v) min(v,[],'omitnan'),T.marginDB,G);
S.powerW = splitapply(@(v) max(v,[],'omitnan'),T.powerW,G);
modes = ["AO","physics","fixedProposal","learnedProposal"];
names = ["AO","Physics only","Fixed proposal","Learned proposal"];
colors = [0.18 0.18 0.18; 0.16 0.39 0.60; 0.67 0.30 0.12; 0.00 0.44 0.36];
markers = {'d','o','s','^'};
loads = unique(S(:,{'K','gammaDB'}),'rows');
budgets = unique(S(S.mode~="AO",{'outer','inner'}),'rows');
assert(~isempty(budgets),'No fast-solver rows found.');
positions = 1:height(budgets);
ticks = compose('%d x %d',budgets.outer,budgets.inner);
F = gobjects(1,3); layouts = gobjects(1,3);
titles = ["Absolute sensing objective","Feasible returned beams","Physical constraint margins"];
for i=1:3
    F(i) = figure('Color','w','Name',titles(i),'Position',[50 80 1050 360*(1+(i~=2))]);
    layouts(i) = tiledlayout(F(i),1+(i~=2),height(loads),'TileSpacing','compact','Padding','compact');
    title(layouts(i),titles(i));
end
% The original experiment uses a common 100 W system power budget.
P = 100;
out = fullfile(folder,'absolute_and_feasibility');
if ~isfolder(out), mkdir(out); end
for c=1:height(loads)
    cond = S(S.K==loads.K(c) & S.gammaDB==loads.gammaDB(c),:);
    caption = sprintf('K = %d, SINR = %g dB',loads.K(c),loads.gammaDB(c));
    fastRows = cell(1,numel(modes));
    data = nan(height(budgets),numel(modes),3);
    for b=1:height(budgets)
        for m=1:numel(modes)
            r = cond(cond.mode==modes(m),:);
            if m~=1, r=r(r.outer==budgets.outer(b) & r.inner==budgets.inner(b),:); end
            fastRows{m}=r;
            if ~isempty(r), data(b,m,3)=100*mean(r.feasible); end
        end
        shared = unique(cond.scene);
        for m=1:numel(modes)
            r=fastRows{m};
            shared=intersect(shared,r.scene(r.feasible & isfinite(r.designDB) & isfinite(r.trueDB)));
        end
        if ~isempty(shared)
            for m=1:numel(modes)
                r=fastRows{m}; r=r(ismember(r.scene,shared),:);
                data(b,m,1)=median(r.designDB);
                data(b,m,2)=median(r.trueDB);
            end
        end
    end
    ylabels = ["Design SCNR (dB)","True-kernel SCNR (dB)","Feasible draws (%)"];
    for q=1:3
        if q==3, ax=nexttile(layouts(2),c);
        else, ax=nexttile(layouts(1),(q-1)*height(loads)+c); end
        hold(ax,'on');
        for m=1:numel(modes)
            % Small horizontal offsets distinguish identical rates and scores.
            plot(ax,positions+(m-2.5)*.055,data(:,m,q),'-','Marker',markers{m}, ...
                'Color',colors(m,:),'LineWidth',1.2,'DisplayName',names(m));
        end
        xticks(ax,positions); xticklabels(ax,ticks); xlim(ax,[.65,height(budgets)+.35]);
        xlabel(ax,'Outer x inner budget'); ylabel(ax,ylabels(q)); title(ax,caption); grid(ax,'on');
        if q==3, ylim(ax,[0 105]); end
        if c==1 && q~=2, legend(ax,'Location','best','FontSize',8); end
    end
    % Residuals at the largest work budget. Include finite margins from failed designs.
    [~,b]=max(budgets.outer.*budgets.inner);
    sceneIDs=unique(cond.scene);
    for q=1:2
        ax=nexttile(layouts(3),(q-1)*height(loads)+c); hold(ax,'on');
        for m=1:numel(modes)
            r=cond(cond.mode==modes(m),:);
            if m~=1, r=r(r.outer==budgets.outer(b) & r.inner==budgets.inner(b),:); end
            [~,index]=ismember(r.scene,sceneIDs);
            if q==1, value=r.marginDB; else, value=P-r.powerW; end
            scatter(ax,index+(m-2.5)*.14,value,22,colors(m,:),markers{m},'DisplayName',names(m));
        end
        if q==1, tol=-1e-4; yname='Worst SINR margin (dB)';
        else, tol=-P*1e-7; yname='Power slack (W)'; end
        yline(ax,tol,'k--','HandleVisibility','off'); grid(ax,'on');
        xlabel(ax,'Channel draw index'); ylabel(ax,yname); title(ax,caption);
        if c==1 && q==1, legend(ax,'Location','best','FontSize',8); end
    end
end
for i=1:3
    files = ["absolute_scnr","physical_feasibility","constraint_margins"];
    exportgraphics(F(i),fullfile(out,files(i)+".pdf"),'ContentType','vector');
end
writetable(S,fullfile(out,'per_scene_objective_and_constraints.csv'));
fprintf('Saved three plots and their per-scene data to %s\n',out);
end
