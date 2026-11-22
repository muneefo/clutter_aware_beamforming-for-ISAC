function F = plot_objective_sweeps(folder, method)
% View the supplied sweeps with no optimizer. For new MATLAB data choose
% "physics", "fixedProposal", "learnedProposal" or "AO" as the second input.
if nargin<1 || isempty(folder), folder=fileparts(mfilename('fullpath')); end
T=readtable(fullfile(folder,'sweep_trials.csv'),'TextType','string');
if nargin<2
    if any(T.method=="joint"), method="joint"; else, method="learnedProposal"; end
end
method=string(method);
assert(any(T.method==method),'Requested method was not run.');
methods=[method,"radar","heuristic","comm"];
labels=[replace(method,"joint","ISAC, 6 LPs"),"Radar only","Heuristic","Comm only"];
palette=[0 .45 .38; .14 .35 .56; .44 .29 .52; .67 .31 .13];
symbols={'o','s','d','^'};
F=gobjects(1,3);
for i=1:2
    if i==1, study="power"; xname='dbm'; xlab='Transmit power (dBm)';
    else, study="sinr"; xname='gamma'; xlab='Communication SINR threshold (dB)'; end
    allStudy=T(T.study==study,:);
    F(i)=figure('Color','white','Name',study+" sweep"); ax=axes('Parent',F(i)); hold(ax,'on');
    users=unique(allStudy.K);
    if i==1, n=numel(methods); else, n=numel(users); end
    for j=1:n
        if i==1
            selected=allStudy(allStudy.method==methods(j),:); label=labels(j);
        else
            selected=allStudy(allStudy.method==method & allStudy.K==users(j),:);
            label="K = "+string(users(j));
        end
        for ker=["perfect","estimated"]
            r=selected(selected.kernel==ker,:);
            if isempty(r), continue; end
            x=unique(r.(xname)); y=nan(size(x));
            for k=1:numel(x)
                use=r.(xname)==x(k) & r.status=="ok" & isfinite(r.trueDB);
                if any(use), y(k)=10*log10(mean(10.^(r.trueDB(use)/10))); end
            end
            if ker=="perfect", line='-'; else, line='--'; end
            name=label+" ("+ker+")"; if i==1 && methods(j)=="comm", name=label; end
            plot(ax,x,y,line,'Color',palette(j,:),'Marker',symbols{j},'LineWidth',1.4,'DisplayName',name);
        end
    end
    xlabel(ax,xlab); ylabel(ax,'True-kernel SCNR (dB)'); grid(ax,'on'); legend(ax,'Location','best');
end
tr=readtable(fullfile(folder,'outer_trace.csv'),'TextType','string');
if ismember('method',tr.Properties.VariableNames), tr=tr(tr.method==method,:); end
assert(~isempty(tr),'No trajectory saved for this method.');
F(3)=figure('Name','Objective per outer update','Color','white'); ax=axes('Parent',F(3)); hold(ax,'on');
keys=["designDB","trueDB"]; labels=["Design objective","True-kernel score"];
if method=="AO", labels(1)="AO covariance objective"; end
iters=unique(tr.iteration); seeds=unique(tr.seed);
for j=1:2
    values=nan(numel(iters),numel(seeds));
    for k=1:numel(seeds)
        r=sortrows(tr(tr.seed==seeds(k),:),'iteration');
        [~,ix]=ismember(r.iteration,iters); values(ix,k)=r.(keys(j));
        % Carry an early terminal iterate forward only for a successful solve.
        success=any(T.study=="trace" & T.method==method & T.seed==seeds(k) & T.status=="ok");
        if success, values(ix(end)+1:end,k)=values(ix(end),k); end
    end
    if any(isfinite(values),'all')
        plot(ax,iters,median(values,2,'omitnan'),'-o','LineWidth',1.4,'DisplayName',labels(j));
    end
end
xlabel(ax,'Completed outer updates'); ylabel(ax,'Median SCNR (dB)'); grid(ax,'on'); legend(ax,'Location','best');
out=fullfile(folder,'matlab_sweep_plots'); if ~isfolder(out), mkdir(out); end
names=["power","sinr","objective"];
for j=1:numel(F), exportgraphics(F(j),fullfile(out,method+"_"+names(j)+".pdf"),'ContentType','vector'); end
fprintf('Saved three figures to %s\n',out);
end
