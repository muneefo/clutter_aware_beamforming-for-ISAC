function out = run_tutorial_comparison(repo,opt)
% Benchmark an unfolded Eq. (124) solver against Fig. 12(b)'s solve_BLP_AO.
% out = run_tutorial_comparison('/path/to/Clutter-Aware-ISAC-Tutorial');
% out = run_tutorial_comparison(repo,struct('nTest',100,'gammaDB',8));
% run_tutorial_comparison(repo,struct('selfTest',true)) needs no CVX or data.
if nargin<1 || isempty(repo)
    [name, folder]=uigetfile('solve_BLP_AO.m','Select the authors'' solve_BLP_AO.m');
    if isequal(name,0), out=[]; return; end
    repo=fileparts(fileparts(fullfile(folder,name)));
end
if nargin<2, opt=struct; end
defaults=struct('depth',3,'nTrain',24,'epochs',3,'nTest',4,'K',4,'gammaDB',8, ...
    'qualityGapDB',0.1,'seed',42,'selfTest',false,'resume',true,'folder','');
keys=fieldnames(opt);
assert(all(ismember(keys,fieldnames(defaults))),'Unknown option.');
for i=1:numel(keys), defaults.(keys{i})=opt.(keys{i}); end
o=defaults;
if isempty(o.folder)
    o.folder=fullfile(repo,sprintf('unfold_K%d_G%g_D%d_T%d_E%d_seed%d', ...
        o.K,o.gammaDB,o.depth,o.nTrain,o.epochs,o.seed));
end
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
selftest();
if o.selfTest, out=true; return; end
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
fprintf('Results folder: %s\n',o.folder);
if o.resume && isfile(checkpoint)
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
f=embedded_fixture();
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

function f=embedded_fixture()
% Independent reference values stored in column-major order.
f.H=reshape(complex([ ...
0.14147131317778738 0.37679662599133501 -1.1693849022306113 -0.19869418857170318 0.33567825859458073 0.47344880667782263 ...
-0.37714176799043841 0.90322969190893876 -0.069918650385121711 -0.083841863900459349 -1.2806743133589917 0.46346411685169869 ...
1.2435183549168842 0.081944064568849617 1.4077928198601406 -0.9406830250229482], ...
[ ...
0.95305609244588219 -1.4645537138516529 -0.13472327618916502 0.047141671571563681 0.37805075811406702 -1.105388056649552 ...
0.40134137691947275 0.72671163502251979 0.60173686923166847 0.113196647861863 -0.31656316944842927 1.536788560147647 ...
0.67793142431390774 0.23298324567177495 -1.2649464319723327 -0.6840113246842543]),[4 2 2]);
f.W=reshape(complex([ ...
-0.062735715468140663 0.090944120484706778 -0.12089233826325876 0.024713144638355618 -0.026445830027501484 -0.057739228790483092 ...
0.066874730146475977 -0.26960673561680193 0.010024379156218 -0.26734952965406095 0.10982669331110056 0.43958185182043263 ...
0.047535330827461017 0.0096247525065476392 0.47748149246292654 -0.01285258974085177 0.0079919191806691724 -0.10518641059835586 ...
-0.092968408235737043 -0.34935573782180213 -0.026890765096931314 -0.13705311482490809 0.090902317797945303 0.35107662784085447 ...
-0.036134819935373215 -0.35740807503838745 0.14805368029799035 0.071456896808196099 -0.0078274167433145855 -0.13295886996248957 ...
0.047182532897381384 0.089186162917569284], ...
[ ...
-0.3258345462133625 -0.068684748217794397 -0.033502295905979265 0.28410226164827423 -0.064904799510957817 -0.34848362590105741 ...
-0.077389204993099717 0.096163064114121274 -0.15869088596513231 0.18926223818843949 0.17609869899240821 0.14117818182177586 ...
0.11592389376246771 0.037002770717444608 -0.098110135969621953 0.17282692744116496 -0.084946622079600201 -0.10999517720067617 ...
0.11647902857674843 0.16064995719325489 0.27784171540808233 0.18372102343303989 0.010099789522071658 0.11458968130710152 ...
0.058117176688211897 -0.078434764157928785 0.096125625708353271 -0.20214529391100608 0.17295119237594678 -0.22349983659444617 ...
0.25850872184472534 -0.13455483434806376]),[4 4 2]);
f.a=reshape(complex([ ...
0.12494934494658232 0.14861987616050082 0.14188853516565331 -0.35251749381208725 -0.094796316891935697 -0.62696660050219866 ...
0.0003386962977188868 0.36753530600637369], ...
[ ...
0.41294790264066411 -0.79529117989860987 0.11001323156180877 -0.05274382915497465 0.60695571307452167 0.18315104477870242 ...
-0.10214065449344992 0.22465967182469057]),[4 2]);
f.b=reshape(complex([ ...
0.42408420419348275 0.13761082090190543 -0.47233457742040197 -0.27293919324741472 -0.088918617085464849 0.14003065312732701], ...
[ ...
-0.36201569062891076 -0.3386921659774666 0.57649644677251854 -0.66356464984162422 -0.66800027571343557 -0.10698915663341627]),[3 2]);
f.R=reshape([ ...
0.10000000000000001 0 0 0 0.10000000000000001 0 ...
0 0 0.10000000000000001 0.14999999999999999 0 0 ...
0 0.14999999999999999 0 0 0 0.14999999999999999],[3 3 2]);
f.eta=reshape([ ...
1.3205562985968082],[1 1]);
f.frac=reshape([ ...
0.47676600097465827],[1 1]);
f.u=reshape(complex([ ...
0.36871375544355722 0.1011956286495521 -0.54865239934290488 -0.23648840101596041 -0.1533450340799036 0.2923094418028036], ...
[ ...
-0.26983190959854675 -0.12738083278635542 0.74542511902969155 -0.84524767373704746 -0.43211653902377467 -0.29306794559133476]),[3 2]);
f.V=reshape(complex([ ...
0.064119364898573075 0.021283329375611674 -0.052642720649762095 0.021283329375611674 0.15926247319575301 0.11534419929469791 ...
-0.052642720649762095 0.11534419929469791 0.17864214516132856 0.080025510979279096 0.011690952192353738 0.048815302532301236 ...
0.03677022181390175 -0.063662384196718907 -0.041789197690766715 -0.12909522977929516 -0.079485114201233192 -0.12171522787821555 ...
-0.02558829299261093 -0.015259884853332458 -0.027764278110007153 0.070531268381190684 -0.062114983454894412 0.093117691057005639 ...
0.14242983325524247 0.037910930160186936 0.093270666905921534 -0.031311060560479623 0.056392028146494355 -0.014242178132329766 ...
0.062199302484167975 0.029024782613109562 0.038823785116189537 0.086461604613721169 -0.02879176357416386 0.012955993071490195 ...
0.080025510979279096 0.03677022181390175 -0.12909522977929516 0.011690952192353738 -0.063662384196718907 -0.079485114201233192 ...
0.048815302532301236 -0.041789197690766715 -0.12171522787821555 0.36875466789193445 0.053638761864520997 0.1736528948635031 ...
0.053638761864520997 0.065347046777381054 0.066905728278946308 0.1736528948635031 0.066905728278946308 0.11675196574242916 ...
-0.22268824075814841 -0.30457512085129346 0.041391404134888302 -0.084403273503681486 -0.048476373349276632 -0.045006597264057605 ...
-0.14136437674851762 -0.12522324628680884 -0.03298724554220045 -0.024664656485430116 0.035925685123788631 0.11999775060017387 ...
-0.037053932040361517 -0.031329458870233355 -0.012659918493083459 -0.047693883708968619 0.0077713582542226645 0.024760127904773683 ...
-0.02558829299261093 0.070531268381190684 0.14242983325524247 -0.015259884853332458 -0.062114983454894412 0.037910930160186936 ...
-0.027764278110007153 0.093117691057005639 0.093270666905921534 -0.22268824075814841 -0.084403273503681486 -0.14136437674851762 ...
-0.30457512085129346 -0.048476373349276632 -0.12522324628680884 0.041391404134888302 -0.045006597264057605 -0.03298724554220045 ...
0.20083447683958822 0.19475795852068351 0.016468931418610479 0.19475795852068351 0.34526192682619772 -0.098848505258382069 ...
0.016468931418610479 -0.098848505258382069 0.099946556086603563 0.043542716509970429 0.02388205672236729 -0.064087965393437474 ...
-0.029128399358802706 0.049857029859017861 -0.14237908628058116 0.064979484216993066 -0.019643791549040508 0.073024772344929267 ...
-0.031311060560479623 0.062199302484167975 0.086461604613721169 0.056392028146494355 0.029024782613109562 -0.02879176357416386 ...
-0.014242178132329766 0.038823785116189537 0.012955993071490195 -0.024664656485430116 -0.037053932040361517 -0.047693883708968619 ...
0.035925685123788631 -0.031329458870233355 0.0077713582542226645 0.11999775060017387 -0.012659918493083459 0.024760127904773683 ...
0.043542716509970429 -0.029128399358802706 0.064979484216993066 0.02388205672236729 0.049857029859017861 -0.019643791549040508 ...
-0.064087965393437474 -0.14237908628058116 0.073024772344929267 0.050271197173280539 -0.023057830688977182 0.032855656151572106 ...
-0.023057830688977182 0.09241368574753564 -0.012103944143785711 0.032855656151572106 -0.012103944143785711 0.089546225076825928 ...
0.25491847607483026 -0.015051288651620568 -0.086723030718147076 -0.015051288651620568 0.078831227619856753 -0.050044527602005076 ...
-0.086723030718147076 -0.050044527602005076 0.11411859858153069 -0.044746813380326829 -0.27823939344959381 0.15894397725568332 ...
-0.1048486466700595 -0.0054041305276961611 0.084615673717821069 0.051097358255082083 0.095817015530167249 -0.12908173355890584 ...
-0.21073808701422592 0.13085059476674765 -0.002372963398051188 0.089537846687496772 -0.0058039723799074502 0.039849512814270703 ...
0.012791398916491267 0.022135901577259788 0.0016790095346645925 0.11142916122535737 -0.04998033546590136 -0.12832706009674288 ...
0.052414504961663312 -0.059376707993791855 0.016377328235415871 -0.046560696337651343 0.18324547205319966 0.028479505660334872 ...
-0.044746813380326829 -0.1048486466700595 0.051097358255082083 -0.27823939344959381 -0.0054041305276961611 0.095817015530167249 ...
0.15894397725568332 0.084615673717821069 -0.12908173355890584 0.20592085726770365 0.10627614374711292 -0.16840612934657501 ...
0.10627614374711292 0.32937888067752247 -0.21596669071436908 -0.16840612934657501 -0.21596669071436908 0.23749981052423341 ...
-0.046245286683831224 -0.063469670302172279 -0.10408213492209877 0.22952480461425279 -0.14198390145980339 -0.04239885006472012 ...
-0.062226773088633835 0.042941887920814957 0.071912702096740139 -0.16807392274034658 -0.002849566328516219 -0.0076941097329672956 ...
-0.18743507357126338 0.04412834241141543 0.11377893159332439 0.18297357104821799 -0.14204453889358504 -0.034533682818278122 ...
-0.21073808701422592 0.089537846687496772 0.012791398916491267 0.13085059476674765 -0.0058039723799074502 0.022135901577259788 ...
-0.002372963398051188 0.039849512814270703 0.0016790095346645925 -0.046245286683831224 0.22952480461425279 -0.062226773088633835 ...
-0.063469670302172279 -0.14198390145980339 0.042941887920814957 -0.10408213492209877 -0.04239885006472012 0.071912702096740139 ...
0.2767183672884076 -0.083032511011531696 0.0060583980046467062 -0.083032511011531696 0.20288284674959514 0.0057169349669708089 ...
0.0060583980046467062 0.0057169349669708089 0.079371698839101709 -0.087825289972580281 -0.018319911298439302 0.080797073281093326 ...
0.049501757341502903 0.17075238346837757 -0.11908302945035126 0.11396208683163742 0.029255974651736461 0.043721996639536492 ...
0.11142916122535737 0.052414504961663312 -0.046560696337651343 -0.04998033546590136 -0.059376707993791855 0.18324547205319966 ...
-0.12832706009674288 0.016377328235415871 0.028479505660334872 -0.16807392274034658 -0.18743507357126338 0.18297357104821799 ...
-0.002849566328516219 0.04412834241141543 -0.14204453889358504 -0.0076941097329672956 0.11377893159332439 -0.034533682818278122 ...
-0.087825289972580281 0.049501757341502903 0.11396208683163742 -0.018319911298439302 0.17075238346837757 0.029255974651736461 ...
0.080797073281093326 -0.11908302945035126 0.043721996639536492 0.2177304879203763 -0.007180176105591074 0.012736628222167113 ...
-0.007180176105591074 0.3941312930459131 -0.024247175053267844 0.012736628222167113 -0.024247175053267844 0.11435175467589734], ...
[ ...
-1.4799787095070411e-19 -0.071210195259197528 -0.037858684152725711 0.071210195259197528 -4.8862401444037012e-19 -0.065828217604647035 ...
0.037858684152725711 0.065828217604647035 -2.9965131458727532e-18 -0.095186499134808422 0.01400430664184675 -0.013659773708451188 ...
0.04293848141380445 0.058805864426586203 0.065595040080401915 0.11439027134900104 0.050223541632774246 0.065003439683640529 ...
0.030541124519608859 0.1244627585409095 -0.070559839529574481 -0.029103042563692609 -0.0025422440072007113 -0.067552283862903489 ...
-0.061656782488646977 -0.1503795575466331 0.023323334779363679 -0.037106728676985848 0.010507610282559147 -0.064431437849788678 ...
-0.04873278559875116 0.015899836990444997 -0.065447352511605328 0.0077057522404660132 -0.039240326199882138 0.011417387671979775 ...
0.095186499134808394 -0.04293848141380445 -0.11439027134900104 -0.01400430664184675 -0.058805864426586203 -0.050223541632774246 ...
0.013659773708451188 -0.065595040080401915 -0.065003439683640529 6.1675998402069401e-19 -0.029148559014992269 0.020277581083840722 ...
0.029148559014992269 -4.9221904623340541e-20 0.01673213465505078 -0.020277581083840722 -0.016732134655050773 -1.3418214314675104e-19 ...
0.036422866945983998 0.1837575647164483 -0.107209398966231 -0.04475113483792164 -0.00055454075669751635 -0.01084533784954874 ...
0.0040443198680278399 0.10076059568407385 -0.05154577624057393 -0.084492866180099907 0.16673677053398539 -0.070157518827669921 ...
-0.016432282969104664 0.012931948499986041 0.02721777084295559 -0.040108595978458617 0.065480238998197271 -0.017727936633223612 ...
-0.030541124519608859 0.029103042563692595 0.061656782488646977 -0.1244627585409095 0.0025422440072007113 0.1503795575466331 ...
0.070559839529574467 0.067552283862903489 -0.023323334779363679 -0.036422866945984005 0.044751134837921633 -0.0040443198680278417 ...
-0.1837575647164483 0.00055454075669751635 -0.10076059568407385 0.107209398966231 0.01084533784954874 0.05154577624057393 ...
4.4122255540415613e-18 -0.071952076048470065 0.024391166694914018 0.071952076048470051 -6.2465106315503954e-19 0.064745154833973598 ...
-0.024391166694914011 -0.064745154833973598 6.5894531518781285e-19 0.031830880349904909 -0.10537421176887604 -0.015278735296135319 ...
0.079335810307783861 -0.15963114638868692 0.0013142146653833883 -0.013219686193037671 0.045039184433840085 -0.0021263048535716846 ...
0.037106728676985848 0.04873278559875116 -0.0077057522404660114 -0.010507610282559147 -0.015899836990445004 0.039240326199882138 ...
0.064431437849788678 0.065447352511605328 -0.011417387671979775 0.084492866180099907 0.016432282969104664 0.040108595978458617 ...
-0.16673677053398539 -0.012931948499986043 -0.065480238998197271 0.070157518827669921 -0.02721777084295559 0.017727936633223612 ...
-0.031830880349904909 -0.079335810307783861 0.013219686193037671 0.10537421176887604 0.15963114638868695 -0.045039184433840085 ...
0.015278735296135326 -0.0013142146653833814 0.0021263048535716854 1.2361001766125279e-18 0.0064429155139443441 0.010469429200356858 ...
-0.0064429155139443441 -6.4456109812101344e-20 -0.082057011726677773 -0.010469429200356858 0.082057011726677773 -2.0343844573407758e-19 ...
7.3044138284891179e-19 -0.10909577123312586 0.097178938790804054 0.10909577123312586 -1.7282758468903321e-19 -0.078940924402696774 ...
-0.097178938790804054 0.078940924402696774 -1.5284962126323547e-18 0.16947528289777508 0.061850626750347526 -0.1821070146885625 ...
0.0088778981953190374 -0.10505812483434189 0.077569540923301955 -0.097488914401158255 0.082707099879969548 0.020434707078377246 ...
-0.060669278867825548 0.083619236388611101 -0.10456084453547501 -0.07486013751914071 0.0067620808284722458 -0.028220964066744344 ...
0.15491557788030846 -0.015544687280253454 0.045156959327155854 -0.16491080334943364 0.16667596253981845 -0.082519469119467878 ...
0.014831255429081584 -0.12356008919501575 -0.054171823485786988 0.018751637501683578 0.0128527230619776 0.04816803289951372 ...
-0.16947528289777508 -0.0088778981953190374 0.097488914401158255 -0.061850626750347526 0.1050581248343419 -0.082707099879969548 ...
0.1821070146885625 -0.077569540923301955 -0.020434707078377246 -1.8853019965887583e-18 0.14871270317454693 -0.091906231218653198 ...
-0.14871270317454693 -3.1710128885738841e-18 0.15094078866499505 0.091906231218653198 -0.15094078866499505 1.0358939262918868e-18 ...
0.075389037016242166 -0.098003550529113817 0.063402711621999702 0.090185380807183732 -0.1509096300753274 0.11259822367060188 ...
-0.1803646216577916 0.12429590665083508 -0.084241799868976466 0.016608364385215318 0.096296899850700515 0.13668121449411402 ...
0.15318974272782138 -0.18858769602733966 0.13558338128398695 -0.045561921060903808 0.021106788478183994 -0.14626755469152025 ...
0.060669278867825548 0.07486013751914071 -0.15491557788030846 -0.083619236388611101 -0.0067620808284722441 0.015544687280253461 ...
0.10456084453547501 0.028220964066744344 -0.045156959327155854 -0.075389037016242166 -0.090185380807183746 0.1803646216577916 ...
0.098003550529113817 0.1509096300753274 -0.12429590665083508 -0.063402711621999702 -0.11259822367060188 0.084241799868976466 ...
4.5929941196254106e-18 -0.12463182187975749 0.033159937381943456 0.12463182187975749 -2.6218804265318021e-18 -0.10030762557900831 ...
-0.033159937381943456 0.10030762557900831 2.7491420001935823e-19 0.097519546428264867 -0.30411867198881604 0.040893527561358861 ...
-0.18900704293266246 0.0033154329288496015 -0.042823083808961632 0.041737658690608442 0.0024492793821803763 -0.079521324758259709 ...
0.16491080334943364 -0.014831255429081591 -0.018751637501683571 -0.16667596253981845 0.12356008919501575 -0.0128527230619776 ...
0.082519469119467878 0.054171823485786988 -0.04816803289951372 -0.016608364385215318 -0.15318974272782135 0.045561921060903815 ...
-0.096296899850700488 0.18858769602733966 -0.02110678847818398 -0.13668121449411402 -0.13558338128398695 0.14626755469152025 ...
-0.097519546428264867 0.18900704293266243 -0.041737658690608435 0.30411867198881604 -0.0033154329288496293 -0.0024492793821803832 ...
-0.040893527561358875 0.042823083808961632 0.079521324758259709 2.9132432562956771e-18 0.087009856928831272 -0.15369411651128825 ...
-0.087009856928831272 4.1252954023491139e-18 0.020566054737184508 0.15369411651128825 -0.020566054737184515 -1.82441366048776e-18]),[9 16 2]);
f.grad=reshape(complex([ ...
0.041340076873128372 -0.078114818321256163 0.09032166266015508 0.067483264320465058 0.088682386904456248 -0.067267504802168171 ...
0.068577620264199751 -0.086417175157220819 0.061540073838326881 -0.17066228489450391 -0.089974526978404593 0.047518953781853793 ...
0.023806726958885787 -0.021560488250047669 -0.11399236281997439 -0.09137828540881418 -0.20838229406374026 0.067743764171220869 ...
0.048770457855889862 -0.017254432090168387 0.037713259307254762 -0.24187274404275538 -0.049135302366490372 0.12383158034653154 ...
-0.11578062661009265 -0.1030161136836916 0.018643612865565921 0.044079838451664438 -0.12363301864709413 -0.088435983134423329 ...
0.03674349076041037 0.061900098871566285], ...
[ ...
0.062549136056826438 0.056347870383171836 -0.061773164239985502 -0.025367414751384453 0.091181786097993781 -0.22142304946476279 ...
-0.003421920938797199 0.022537176134327593 -0.15947162813449053 0.22465600152955095 -0.10720415158762527 0.050576759013995498 ...
-0.089341273367775351 -0.046485782145956611 0.0056430986735749814 0.050343585086615406 -0.097656831844002673 -0.11530986436194068 ...
-0.073222325097571914 0.014304135546192191 0.33360098840733399 0.16563698466558094 0.020891999616653393 0.027080962532302244 ...
0.18106356928478712 -0.0053112676898044442 -0.033677945695657213 0.054065485487070519 0.048060219312261225 0.023153611081825353 ...
-0.020505525398602412 0.044566027927997097]),[4 4 2]);
f.sinr=reshape([ ...
0.0061392392067190424 0.043564764822523548 0.40974402238468133 0.30277873760204765],[2 2]);
end
