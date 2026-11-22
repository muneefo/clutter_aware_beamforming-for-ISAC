function [fig,summary] = plot_learned_sinr(folder)
% Compare learned and fixed proposals over the 50 dBm SINR sweep.
% folder = run_objective_sweeps(repo,outWU);
% [fig,summary] = plot_learned_sinr(folder);
if nargin<1
    folder=uigetdir(pwd,'Select the MATLAB sweep folder containing sweep_trials.csv');
    if isequal(folder,0), fig=[]; summary=table(); return; end
end
source=fullfile(folder,'sweep_trials.csv');
assert(isfile(source),'No sweep_trials.csv in %s.',folder);
T=readtable(source,'TextType','string');
keys={'K','gamma','kernel','seed'};
T=T(T.study=="sinr" & T.dbm==50,:);
F=T(T.method=="fixedProposal",:);
L=T(T.method=="learnedProposal",:);
assert(~isempty(F) && ~isempty(L), ...
    'This CSV needs fixedProposal and learnedProposal rows. Select the MATLAB model sweep.');
assert(height(unique(F(:,keys)))==height(F) && height(unique(L(:,keys)))==height(L), ...
    'Multiple records for a method and channel at one operating point.');
F=sortrows(F,keys); L=sortrows(L,keys);
assert(isequal(F(:,keys),L(:,keys)),'The attempted fixed and learned draws differ.');
okF=F.status=="ok" & isfinite(F.trueDB);
okL=L.status=="ok" & isfinite(L.trueDB);
paired=okF & okL;
[G,summary]=findgroups(F(:,{'K','gamma','kernel'}));
summary.attempted=splitapply(@numel,F.seed,G);
summary.fixedSuccessful=splitapply(@sum,okF,G);
summary.learnedSuccessful=splitapply(@sum,okL,G);
summary.paired=splitapply(@sum,paired,G);
f=F.trueDB; f(~paired)=NaN;
l=L.trueDB; l(~paired)=NaN;
% Average the linear SCNR for the absolute curves, then convert to dB.
meanSCNR=@(db) 10*log10(mean(10.^(db/10),'omitnan'));
summary.fixedDB=splitapply(meanSCNR,f,G);
summary.learnedDB=splitapply(meanSCNR,l,G);
% This statistic compares each learned design to its own fixed control.
summary.pairedMedianGainDB=splitapply(@(v)median(v,'omitnan'),l-f,G);
users=unique(summary.K); kernels=["perfect","estimated"];
pal=[0.10 0.36 0.56; 0.04 0.43 0.36; 0.48 0.26 0.52];
fig=figure('Color','w','Name','Learned SINR sweep','Position',[100 100 960 620]);
panels=tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
for i=1:2
    a=nexttile(panels); hold(a,'on');
    b=nexttile(panels); hold(b,'on');
    yline(b,0,'k:','HandleVisibility','off');
    for j=1:numel(users)
        r=sortrows(summary(summary.K==users(j) & summary.kernel==kernels(i),:),'gamma');
        if isempty(r), continue; end
        c=pal(1+mod(j-1,size(pal,1)),:);
        label=sprintf('K = %d',users(j));
        % Open circles let the learned curve remain visible when the two coincide.
        plot(a,r.gamma,r.fixedDB,'--s','Color',c,'MarkerSize',4,'LineWidth',1.0, ...
            'DisplayName',[label ', fixed']);
        plot(a,r.gamma,r.learnedDB,'-o','Color',c,'MarkerSize',7,'LineWidth',1.2, ...
            'DisplayName',[label ', learned']);
        plot(b,r.gamma,r.pairedMedianGainDB,'-o','Color',c,'LineWidth',1.2,'DisplayName',label);
    end
    title(a,kernels(i)+" clutter kernel"); ylabel(a,'True-kernel SCNR (dB)');
    title(b,kernels(i)+" clutter kernel"); ylabel(b,'Median learned minus fixed (dB)');
    for ax=[a b], grid(ax,'on'); xlabel(ax,'Communication SINR threshold (dB)'); end
    if i==1, legend(a,'Location','best','NumColumns',2); legend(b,'Location','best'); end
end
pdf=fullfile(folder,'learned_vs_fixed_sinr.pdf');
exportgraphics(fig,pdf,'ContentType','vector');
writetable(summary,fullfile(folder,'learned_vs_fixed_sinr.csv'));
fprintf('Saved %s\n',pdf);
end
