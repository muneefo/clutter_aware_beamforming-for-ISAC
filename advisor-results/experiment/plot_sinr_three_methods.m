function [fig,S] = plot_sinr_three_methods(folder)
% Plot the 50 dBm sweep from run_objective_sweeps (authorsAO=true)
% or run_sinr_three_methods. One panel per user count and design kernel.
T=readtable(fullfile(folder,'sweep_trials.csv'),'TextType','string');
keys={'K','gamma','kernel','seed'};
methods=["AO","fixedProposal","learnedProposal"];
labels=["AO (SDP)","Fixed proposal","Learned proposal"];
assert(all(ismember([keys {'study','method','dbm','status','trueDB'}],T.Properties.VariableNames)), ...
    'Select the sweep CSV folder, not the single-setting W/U trials folder.');
T=T(T.study=="sinr" & T.dbm==50 & ismember(T.method,methods),:);
assert(all(ismember(methods,T.method)), ...
    'This sweep needs AO, fixedProposal and learnedProposal rows. Use run_sinr_three_methods(repo,outWU).');
pair=cell(1,3);
for j=1:3
    pair{j}=sortrows(T(T.method==methods(j),:),keys);
    assert(height(unique(pair{j}(:,keys)))==height(pair{j}),'Repeated method/condition/seed.');
    assert(isequal(pair{1}(:,keys),pair{j}(:,keys)),'The three methods have different attempted draws.');
end
[G,S]=findgroups(pair{1}(:,keys(1:3)));
ok=false(height(pair{1}),3);
for j=1:3
    r=pair{j}; ok(:,j)=r.status=="ok" & isfinite(r.trueDB);
end
shared=all(ok,2);
S.attempted=splitapply(@numel,pair{1}.seed,G);
S.paired=splitapply(@sum,shared,G);
for j=1:3
    v=pair{j}.trueDB; v(~shared)=NaN;
    S.(methods(j)+"_dB")=splitapply(@(x)10*log10(mean(10.^(x/10),'omitnan')),v,G);
    S.(methods(j)+"_usable")=splitapply(@sum,ok(:,j),G);
end
users=unique(S.K); kernels=unique(S.kernel,'stable');
fig=figure('Color','w','Name','SCNR versus required SINR','Position',[80 100 360*numel(users) 270*numel(kernels)]);
tiles=tiledlayout(fig,numel(kernels),numel(users),'Padding','compact','TileSpacing','compact');
sty={'-d','--s',':o'}; sz=[5 5 8]; colors=[.15 .15 .15; .16 .38 .66; .72 .28 .08];
for i=1:numel(kernels)
    for k=1:numel(users)
        ax=nexttile(tiles); hold(ax,'on');
        r=sortrows(S(S.K==users(k) & S.kernel==kernels(i),:),'gamma');
        for j=1:3
            plot(ax,r.gamma,r.(methods(j)+"_dB"),sty{j},'Color',colors(j,:), ...
                'LineWidth',1.4,'MarkerSize',sz(j),'MarkerFaceColor','none','DisplayName',labels(j));
        end
        title(ax,sprintf('K = %d, %s clutter',users(k),char(kernels(i))));
        xticks(ax,unique(T.gamma)); grid(ax,'on'); box(ax,'on');
        xlabel(ax,'Required communication SINR (dB)');
        if k==1, ylabel(ax,'True sensing SCNR (dB)'); end
        if i==1 && k==1, legend(ax,'Location','best'); end
    end
end
exportgraphics(fig,fullfile(folder,'scnr_vs_sinr_three_methods.pdf'),'ContentType','vector');
writetable(S,fullfile(folder,'scnr_vs_sinr_three_methods.csv'));
fprintf('Figure and paired counts saved to %s\n',folder);
end
