function out = compare_tutorial_unfolding(repo,opt)
% Benchmark an unfolded Eq. (124) solver against Fig. 12(b)'s solve_BLP_AO.
% out = compare_tutorial_unfolding('/path/to/Clutter-Aware-ISAC-Tutorial');
% out = compare_tutorial_unfolding(repo,struct('nTest',100,'gammaDB',8));
% compare_tutorial_unfolding(repo,struct('selfTest',true)) needs no CVX or data.
if nargin<2, opt=struct; end
defaults=struct('depth',3,'nTrain',24,'epochs',3,'nTest',4,'K',4,'gammaDB',8, ...
    'qualityGapDB',0.1,'seed',42,'selfTest',false,'resume',false,'folder','tutorial_unfold_results');
keys=fieldnames(opt);
assert(all(ismember(keys,fieldnames(defaults))),'Unknown option.');
for i=1:numel(keys), defaults.(keys{i})=opt.(keys{i}); end
o=defaults;
for name={'depth','nTrain','epochs','nTest','K'}
    validateattributes(o.(name{1}),{'numeric'},{'scalar','integer','positive'});
end
validateattributes(o.seed,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(o.qualityGapDB,{'numeric'},{'scalar','real','finite','nonnegative'});
validateattributes(o.gammaDB,{'numeric'},{'scalar','real','finite'});
saveRng=rng; restoreRng=onCleanup(@()rng(saveRng)); %#ok<NASGU>
oldPath=path; restorePath=onCleanup(@()path(oldPath)); %#ok<NASGU>
assert(isfolder(fullfile(repo,'function')),'Pass the root folder of the authors'' repository.');
addpath(fullfile(repo,'function'));
if o.selfTest, selftest(); out=true; return; end
assert(exist('cvx_begin','file')==2,'Set up CVX and a working SDP solver first.');
file=fullfile(repo,'data','covMat_BLP.mat');
assert(isfile(file),'Generate data/covMat_BLP.mat with the authors'' scripts first.');
s=load(file,'para','Vcc','Vcc_hat','R_eta'); p=s.para;
p.N=32; p.L=64; p.Pt=100; p.Ku=o.K;
assert(p.Nt==16 && p.Nr==16 && p.Ns==16 && o.K<p.Ns,'Expected Fig. 12(b) dimensions with K < 16.');
assert(numel(s.Vcc_hat)>=p.N && numel(s.Vcc)>=p.N && size(s.R_eta,3)>=p.N,'Missing subcarriers.');
d.p=p; d.V=s.Vcc_hat(1:p.N); d.R=s.R_eta(:,:,1:p.N);
trueKernel=s.Vcc(1:p.N); clear s
d.noise=1e-19*p.deltaf; d.gamma=10^(o.gammaDB/10)*ones(p.N,o.K);
d.opt=struct('maxAO',30,'maxDinkel',30,'tolAO_dB',.01, ...
    'tolDinkel_eta',1e-3,'lambdaRidge',0,'verbose',0);
f=p.fc+(0:p.N-1)*p.deltaf;
d.sig=p.Pt*(p.c0./f).^2*1e4*10^.5/((4*pi)^3*41.8^4);
for n=1:p.N
    d.a{n,1}=exp(-1i*pi*(0:p.Nt-1)'*sind(-10)*f(n)/p.fc)/sqrt(p.Nt);
    d.b{n,1}=exp(-1i*pi*(0:p.Nr-1)'*sind(-10)*f(n)/p.fc)/sqrt(p.Nr);
end
if ~isfolder(o.folder), mkdir(o.folder); end
% Channel realizations and random initial beams use independent streams.
% Training, warm-up, and test splits are disjoint offsets for a given run.
stride=max(o.nTrain,o.nTest)+10;
train0=o.seed; warm0=train0+stride; test0=warm0+stride;
checkpoint=fullfile(o.folder,'trained_model.mat');
if o.resume
    assert(isfile(checkpoint),'Missing trained_model.mat. Use the original run folder or set opt.folder.');
    ck=load(checkpoint,'model','trainInfo','o');
    for key={'depth','nTrain','epochs','K','gammaDB','seed'}
        name=key{1};
        assert(isequal(ck.o.(name),o.(name)),'Training option %s differs from the checkpoint.',name);
    end
    model=ck.model; trainInfo=ck.trainInfo;
    fprintf('Resumed trained policy from %s.\n', checkpoint);
else
    clock=tic; [model,trainInfo]=trainmodel(d,o,train0); trainInfo.seconds=toc(clock);
    save(checkpoint,'model','trainInfo','o');
end
fprintf('Fixed control step %.3g; %d/%d training scenes had a feasible anchor.\n', ...
    model.fixed,trainInfo.usable,o.nTrain);
warm=channels(d,warm0);
rng(beamseed(warm0),'twister');
% Profile a separate, untimed AO run. TotalTime includes child calls.
% This first run also checks the installed CVX backend before the timed trials.
profile clear
profile on
try
    baseline(warm);
catch err
    profile off
    rethrow(err);
end
profile off
prof=profile('info'); functions=prof.FunctionTable;
ft=table(string({functions.FunctionName}).',[functions.NumCalls].',[functions.TotalTime].', ...
    'VariableNames',{'FunctionName','NumCalls','TotalTime'});
wanted=endsWith(string(ft.FunctionName),["solve_BLP_AO","solve_TX_Dinkelbach","mvdr_beamformer","recover_beams"]);
baselineProfile=ft(wanted,{'FunctionName','NumCalls','TotalTime'});
writetable(baselineProfile,fullfile(o.folder,'baseline_profile.csv'));
rollout(warm,model,NaN,o.depth,[],false,beamseed(warm0));
[trialTemplate,stageTemplate]=result_templates();
rows=repmat(trialTemplate,0,1); stages=repmat(stageTemplate,0,1);
for i=1:o.nTest
    seed=test0+i; test=channels(d,seed); initSeed=beamseed(seed);
    methods=["AO_SDP","fixed","learned"]; order=circshift(1:3,mod(i-1,3));
    for k=order
        rec=trialTemplate; rec.scene=seed; rec.method=methods(k);
        rng(initSeed,'twister'); timer=tic;
        try
            if k==1
                v=baseline(test); rec.seconds=toc(timer);
                W=cat(2,v.w_comm,v.Ws); u=v.u;
                rec.iterations=numel(v.scnr_hist)-1;
                rec.paperCovTrueSCNRdB=dbscore(v.RX,u,test,trueKernel);
            else
                step=NaN; if k==2, step=model.fixed; end
                v=rollout(test,model,step,o.depth,[],true,initSeed);
                rec.seconds=toc(timer); W=v.W; u=v.u; rec.iterations=o.depth;
                for j=1:o.depth
                    beam=v.history{j};
                    [ok,~,~]=audit(test,beam.W);
                    st=stageTemplate; st.scene=seed; st.method=methods(k); st.layer=j;
                    st.seconds=v.times(j); st.feasible=ok;
                    st.trueSCNRdB=dbscore(covs(beam.W),beam.u,test,trueKernel);
                    stages(end+1,1)=st; %#ok<AGROW>
                end
            end
            [rec.feasible,rec.minSINRmarginDB,rec.powerW]=audit(test,W);
            Q=covs(W);
            rec.trueSCNRdB=dbscore(Q,u,test,trueKernel);
            rec.designSCNRdB=dbscore(Q,u,test,test.V);
            if ~rec.feasible, rec.message="Returned beams violate power or SINR"; end
        catch err
            if isnan(rec.seconds), rec.seconds=toc(timer); end
            rec.feasible=false; rec.message=string(err.message);
            warning('%s, scene %d: %s',methods(k),seed,err.message);
        end
        rows(end+1,1)=rec; %#ok<AGROW>
    end
    writetable(struct2table(rows),fullfile(o.folder,'trials.csv'));
    fprintf('Paired test scene %d of %d complete.\n',i,o.nTest);
end
T=struct2table(rows); L=struct2table(stages);
if ~isempty(L), writetable(L,fullfile(o.folder,'layers.csv')); end
T.gapToAOdB=nan(height(T),1); T.firstHitSeconds=T.gapToAOdB; T.speedAtQuality=T.gapToAOdB;
for i=1:o.nTest
    ids=find(T.scene==test0+i); ref=ids(T.method(ids)=="AO_SDP");
    if ~T.feasible(ref), continue; end
    for r=ids(:).'
        if ~T.feasible(r), continue; end
        T.gapToAOdB(r)=T.trueSCNRdB(ref)-T.trueSCNRdB(r);
        if r==ref
            T.firstHitSeconds(r)=T.seconds(ref);
        elseif ~isempty(L)
            hits=L.scene==T.scene(r) & L.method==T.method(r) & L.feasible ...
                & L.trueSCNRdB>=T.trueSCNRdB(ref)-o.qualityGapDB;
            if any(hits), T.firstHitSeconds(r)=min(L.seconds(hits)); end
        end
        T.speedAtQuality(r)=T.seconds(ref)/T.firstHitSeconds(r);
    end
end
writetable(T,fullfile(o.folder,'trials.csv'));
out=struct('trials',T,'layers',L,'model',model,'training',trainInfo,'options',o);
out.baselineProfile=baselineProfile;
for k=1:numel(methods)
    use=T.method==methods(k); valid=use & T.feasible;
    fprintf('%s: feasible %d/%d, median feasible true SCNR %.3f dB, median attempted design time %.4f s, quality target hits %d.\n', ...
        methods(k),nnz(valid),nnz(use),median(T.trueSCNRdB(valid),'omitnan'), ...
        median(T.seconds(use),'omitnan'),nnz(isfinite(T.firstHitSeconds(use))));
end
disp(T(:,{'scene','method','seconds','feasible','trueSCNRdB','gapToAOdB','speedAtQuality'}));
end

function [trial,stage]=result_templates()
trial=struct('scene',NaN,'method',"",'seconds',NaN,'feasible',false, ...
            'trueSCNRdB',NaN,'designSCNRdB',NaN,'paperCovTrueSCNRdB',NaN, ...
            'minSINRmarginDB',NaN,'powerW',NaN,'iterations',NaN,'message',"");
stage=struct('scene',NaN,'method',"",'layer',NaN,'seconds',NaN, ...
    'feasible',false,'trueSCNRdB',NaN);
end

function j=beamseed(seed)
j=mod(17389+1664525*double(seed),2^32);
end

function x=channels(d,seed)
rng(seed,'twister'); x=d; p=x.p;
x.H=(randn(p.Nt,p.N,p.Ku)+1i*randn(p.Nt,p.N,p.Ku))/sqrt(2);
dist=randi([50,400],1,1,p.Ku);
x.H=x.H.*10.^((-30-27*log10(dist))/20);
end

function v=baseline(x)
v=solve_BLP_AO(x.p,x.sig,x.V,x.R,x.a,x.b,x.H,x.gamma,x.noise,x.opt);
end

function Q=covs(W)
Q=pagemtimes(W,'none',W,'ctranspose');
end

function y=dbscore(Q,u,x,kernel)
s=compute_SCNR(Q,u,kernel,x.R,x.sig,x.a,x.b);
assert(isreal(s) && isfinite(s) && s>0,'Invalid SCNR');
y=10*log10(s);
end

function [ok,margin,power]=audit(x,W)
ratios=sinrs(x,W)./x.gamma;
margin=10*log10(min(ratios(:))); power=sum(abs(W(:)).^2);
ok=all(isfinite(W(:))) && isfinite(margin) && margin>=-1e-4 && power<=x.p.Pt*(1+1e-7);
end

function ratios=sinrs(x,W)
p=x.p; ratios=zeros(p.N,p.Ku);
for n=1:p.N
    H=reshape(x.H(:,n,:),p.Nt,p.Ku);
    y=abs(H'*W(:,:,n)).^2;
    desired=diag(y(:,1:p.Ku));
    y(1:p.Ku+1:p.Ku^2)=0;
    ratios(n,:)=(desired./(sum(y,2)+x.noise)).';
end
end

function W=alignphases(W,x)
for n=1:x.p.N
    H=reshape(x.H(:,n,:),x.p.Nt,x.p.Ku);
    ang=angle(diag(H'*W(:,1:x.p.Ku,n)));
    W(:,1:x.p.Ku,n)=W(:,1:x.p.Ku,n).*exp(-1i*ang.');
end
end

function [anchor,W]=start(x,seed)
% The original solver starts at random transmit covariance, before QoS.
% Replay that draw, then repair it before the first beam-space update.
rng(seed,'twister'); p=x.p;
raw=(randn(p.Nt,p.Ns,p.N)+1i*randn(p.Nt,p.Ns,p.N))/sqrt(2);
raw=raw*(sqrt(p.Pt)/norm(raw(:)));
anchor=zeros(size(raw)); S=zeros(size(raw));
for n=1:p.N
    H=reshape(x.H(:,n,:),p.Nt,p.Ku);
    [U,R]=qr(H,0);
    Z=U/R';  % H'*Z = I
    anchor(:,1:p.Ku,n)=Z.*sqrt(x.gamma(n,:)*x.noise);
    S(:,p.Ku+1:end,n)=raw(:,p.Ku+1:end,n)-U*(U'*raw(:,p.Ku+1:end,n));
end
need=sum(abs(anchor(:)).^2);
if need>=p.Pt, error('unfold:Anchor','ZF anchor exceeds the power budget'); end
% Reserve half the surplus for QoS margin and half for sensing streams.
anchor=anchor*sqrt((p.Pt+need)/(2*need))+S*(sqrt((p.Pt-need)/2)/norm(S(:)));
assert(audit(x,anchor),'Infeasible numerical anchor.');
W=repair(x,raw,anchor);
end

function W=repair(x,Y,anchor)
% Convex interpolation after phase alignment. Bisection finds a feasible
% point on this segment, rather than a Euclidean projection.
Y=alignphases(Y,x);
Y=Y/max(1,norm(Y(:))/sqrt(x.p.Pt));
if all(sinrs(x,Y)>=x.gamma,'all'), W=Y; return; end
lo=0; hi=1;
for j=1:23
    t=(lo+hi)/2;
    if all(sinrs(x,(1-t)*anchor+t*Y)>=x.gamma,'all'), lo=t; else, hi=t; end
end
W=(1-lo)*anchor+lo*Y;
end

function [eta,u,G,frac]=physics(x,W)
p=x.p; Q=covs(W); u=zeros(p.Nr,p.N); G=zeros(size(W));
target=0; disturbance=0; cold=0;
for n=1:p.N
    C=kernel_apply(x.V{n},Q(:,:,n),p.Nr);
    R=C+x.R(:,:,n); b=x.b{n}; a=x.a{n};
    v=mvdr_beamformer(R,b,0); u(:,n)=v;
    target=target+x.sig(n)*abs(v'*b)^2*real(a'*Q(:,:,n)*a);
    disturbance=disturbance+real(v'*R*v); cold=cold+real(v'*C*v);
end
eta=target/disturbance; frac=cold/disturbance;
assert(isfinite(eta) && eta>0 && disturbance>0,'Invalid design objective.');
for n=1:p.N
    v=u(:,n); a=x.a{n}; A=x.sig(n)*abs(v'*x.b{n})^2*(a*a');
    G(:,:,n)=(A-eta*build_Ac(x.V{n},v,p.Nt))*W(:,:,n);
end
end

function v=rollout(x,net,fixed,D,perturb,record,seed)
timer=tic;
[anchor,W]=start(x,seed);
v.history=cell(1,D); v.times=nan(1,D); v.J=zeros(numel(net.w),D);
[eta,u,G,frac]=physics(x,W);
for j=1:D
    margin=min(sinrs(x,W)./x.gamma,[],'all');
    feat=[frac; tanh(log(max(margin,eps))); j/D];
    if isnan(fixed)
        if record
            logit=policy(net.w,feat);
        else
            [logit,v.J(:,j)]=policy(net.w,feat);
        end
        if ~isempty(perturb), logit=logit+perturb(j); end
        step=net.lo+(net.hi-net.lo)/(1+exp(-logit));
    else
        step=fixed;
    end
    sizeG=norm(G(:));
    if sizeG>0
        Y=W+step*sqrt(x.p.Pt)*G/sizeG;
        Y=repair(x,Y,anchor);
        [next,vNext,gNext,fNext]=physics(x,Y);
        if next>=eta
            W=Y; u=vNext; G=gNext; frac=fNext; eta=next;
        end
    end
    if record
        v.history{j}=struct('W',W,'u',u);
        v.times(j)=toc(timer);
    end
end
v.W=W; v.u=u; v.logSCNR=log(eta);
end

function [z,J]=policy(w,f)
% Shared 3 -> 8 tanh -> 1 logit, including biases (41 parameters).
A=reshape(w(1:24),8,3); b=w(25:32); c=w(33:40);
h=tanh(A*f+b); z=c'*h+w(41);
if nargout>1
    dh=c.*(1-h.^2); dA=dh*f'; J=[dA(:);dh;h;1];
end
end

function [net,info]=trainmodel(d,o,offset)
rng(19,'twister'); net.w=[.15*randn(24,1);zeros(8,1);.1*randn(8,1);0];
net.lo=.01; net.hi=1.0;
grid=[.03,.1,.3,.6,1.0]; total=zeros(size(grid)); usable=[];
for i=1:o.nTrain
    seed=offset+i; x=channels(d,seed);
    try
        score=zeros(size(grid));
        for k=1:numel(grid)
            v=rollout(x,net,grid(k),o.depth,[],false,beamseed(seed)); score(k)=v.logSCNR;
        end
    catch err
        if strcmp(err.identifier,'unfold:Anchor'), continue; end
        rethrow(err);
    end
    usable(end+1)=seed; total=total+score; %#ok<AGROW>
end
assert(~isempty(usable),'No usable training scenes.');
[~,best]=max(total); net.fixed=grid(best);
t=min(.98,max(.02,(net.fixed-net.lo)/(net.hi-net.lo))); net.w(end)=log(t/(1-t));
m=zeros(size(net.w)); s=m; iter=0; epsFD=.005;
for epoch=1:o.epochs
    rng(offset+epoch,'twister'); order=usable(randperm(numel(usable)));
    for seed=order
        x=channels(d,seed); bs=beamseed(seed);
        v=rollout(x,net,NaN,o.depth,[],false,bs);
        % Total downstream sensitivity to each layer logit; future layers
        % react normally to the perturbed state. The MLP Jacobian is analytic.
        adj=zeros(o.depth,1);
        for j=1:o.depth
            delta=zeros(1,o.depth); delta(j)=epsFD;
            vp=rollout(x,net,NaN,o.depth,delta,false,bs);
            vm=rollout(x,net,NaN,o.depth,-delta,false,bs);
            adj(j)=(vp.logSCNR-vm.logSCNR)/(2*epsFD);
        end
        g=v.J*adj; g=g/max(1,norm(g)); iter=iter+1;
        m=.9*m+.1*g; s=.999*s+.001*g.^2;
        net.w=net.w+.01*(m/(1-.9^iter))./(sqrt(s/(1-.999^iter))+1e-8);
    end
end
info=struct('usable',numel(usable),'trainingSeeds',usable,'grid',grid, ...
    'meanFixedLogSCNR',total/numel(usable),'optimizer',"Adam ascent", ...
    'updates',iter,'logitDifference',epsFD);
end

function selftest()
[trial,stage]=result_templates();
rows=repmat(trial,0,1); stages=repmat(stage,0,1);
for idx=1:2
    trial.scene=idx; stage.layer=idx;
    rows(end+1,1)=trial; stages(end+1,1)=stage;
end
T=struct2table(rows); L=struct2table(stages);
assert(isequal(T.scene,[1;2]) && isequal(L.layer,[1;2]),'Result recording failed.');
% Numerical oracle generated with NumPy from explicit scatterer matrices.
f=load(fullfile(fileparts(mfilename('fullpath')),'beam_checks.mat'));
p=struct('Nt',4,'Nr',3,'Ns',4,'N',2,'Ku',2,'Pt',2);
x=struct('p',p,'noise',.01,'gamma',ones(2),'sig',[.8 1.2]);
x.R=f.R; x.H=f.H;
for n=1:p.N
    x.a{n,1}=f.a(:,n); x.b{n,1}=f.b(:,n); x.V{n,1}=f.V(:,:,n);
end
[val,u,g,frac]=physics(x,f.W);
assert(abs(val-f.eta)<1e-10 && abs(frac-f.frac)<1e-10,'SCNR or clutter fraction mismatch');
assert(norm(u(:)-f.u(:))<1e-9 && norm(g(:)-f.grad(:))<1e-9,'MVDR or gradient mismatch');
assert(abs(val-compute_SCNR(covs(f.W),u,x.V,x.R,x.sig,x.a,x.b))<1e-10,'SCNR convention mismatch');
s=sinrs(x,f.W); assert(max(abs(s-f.sinr),[],'all')<1e-10,'SINR mismatch');
net=struct('w',linspace(-.2,.3,41).','lo',.01,'hi',1);
v=rollout(x,net,NaN,3,[],true,21);
prev=-inf;
for j=1:3
    z=v.history{j}; assert(audit(x,z.W),'Infeasible layer.');
    score=dbscore(covs(z.W),z.u,x,x.V);
    assert(score>=prev-1e-9,'Decreasing design objective'); prev=score;
end
% Independently check the end-to-end derivative along a parameter direction.
v=rollout(x,net,NaN,3,[],false,21);
epsFD=3e-4; adj=zeros(3,1);
for j=1:3
    e=zeros(1,3); e(j)=epsFD;
    vp=rollout(x,net,NaN,3,e,false,21); vm=rollout(x,net,NaN,3,-e,false,21);
    adj(j)=(vp.logSCNR-vm.logSCNR)/(2*epsFD);
end
d=cos((1:41).'); d=d/norm(d);
np=net; nm=net; np.w=np.w+epsFD*d; nm.w=nm.w-epsFD*d;
vp=rollout(x,np,NaN,3,[],false,21); vm=rollout(x,nm,NaN,3,[],false,21);
fd=(vp.logSCNR-vm.logSCNR)/(2*epsFD);
assert(abs(fd-d'*v.J*adj)<.005*(1+abs(fd)),'Policy sensitivity mismatch');
disp('Reference matrix checks, feasible stages and policy derivative check passed.');
end
