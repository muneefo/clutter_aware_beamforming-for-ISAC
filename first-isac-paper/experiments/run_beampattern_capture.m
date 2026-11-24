function out = run_beampattern_capture(repo, checkpoint, user)
% Reproduce one paper scene, retain the beamformers, and plot their patterns.
% Seed 10009 is the lower-index scene adjacent to the median 15-allocation
% learned-policy SCNR gap for K=4 and an 8 dB SINR requirement.
% Example:
% out = run_beampattern_capture(repo, trainedModelFile);
if nargin<3, user=struct(); end
if isstruct(checkpoint) && isfield(checkpoint,'model') && isfield(checkpoint,'options')
    model=checkpoint.model; training=checkpoint.options;
elseif isstruct(checkpoint) && isfield(checkpoint,'manifest')
    model=checkpoint.manifest.model; training=checkpoint.manifest.training;
else
    saved=load(checkpoint);
    if isfield(saved,'manifest')
        model=saved.manifest.model; training=saved.manifest.training;
    else
        model=saved.model; training=saved.o;
    end
end
cfg=struct('seeds',10009,'conditions',[4 8], ...
    'budgets',[5 3],'repeats',1,'folder',fullfile(repo,'beampattern_scene10009'));
names=fieldnames(user);
assert(all(ismember(names,fieldnames(cfg))),'Unknown option.');
for i=1:numel(names), cfg.(names{i})=user.(names{i}); end
validateattributes(cfg.seeds,{'numeric'},{'integer','vector','nonempty','positive','finite'});
assert(numel(unique(cfg.seeds))==numel(cfg.seeds),'Duplicate seeds.');
validateattributes(cfg.repeats,{'numeric'},{'integer','scalar','positive'});
assert(size(cfg.budgets,2)==2 && all(cfg.budgets(:)>=1 & mod(cfg.budgets(:),1)==0));
assert(size(cfg.conditions,2)==2 && all(isfinite(cfg.conditions(:))) && ...
    all(cfg.conditions(:,1)>=1 & mod(cfg.conditions(:,1),1)==0 & cfg.conditions(:,1)<16));
assert(size(unique(cfg.conditions,'rows'),1)==size(cfg.conditions,1) && ...
    size(unique(cfg.budgets,'rows'),1)==size(cfg.budgets,1),'Duplicate settings.');
trainSeeds=training.seed+(1:training.nTrain);
oldTest=training.seed+2*(max(training.nTrain,training.nTest)+10)+(1:training.nTest);
assert(isempty(intersect(cfg.seeds,[trainSeeds,oldTest,111:114])), 'Use fresh channel seeds.');
state=rng; protect=onCleanup(@()rng(state)); %#ok<NASGU>
oldPath=path; restore=onCleanup(@()path(oldPath)); %#ok<NASGU>
addpath(fullfile(repo,'function'));
covFile=fullfile(repo,'data','covMat_BLP.mat');
data=load(covFile,'para','Vcc_hat','Vcc','R_eta','tarPara', ...
    'theta_cc_all','sigma2_cc_all');
backend=choose_backend();
if ~isfolder(cfg.folder), mkdir(cfg.folder); end
info=dir(covFile); manifest=struct('config',cfg,'model',model,'training',training, ...
    'backend',backend,'matlab',version,'platform',computer,'repo',repo, ...
    'covBytes',info.bytes,'covDate',info.datenum,'driver',fileread([mfilename('fullpath'),'.m']), ...
    'aoCode',fileread(which('solve_BLP_AO')),'txCode',fileread(which('solve_TX_Dinkelbach')));
meta=fullfile(cfg.folder,'manifest.mat');
if isfile(meta)
    old=load(meta,'manifest');
    assert(isequaln(old.manifest,manifest),'Run settings or inputs changed. Select another output folder.');
else
    save(meta,'manifest');
end
% A complete scene/condition is the resume unit. Interruption during a scene
% repeats only that unit. Timing repeats are observations within one channel.
cells=cell(numel(cfg.seeds),size(cfg.conditions,1));
for c=1:size(cfg.conditions,1)
    K=cfg.conditions(c,1); threshold=cfg.conditions(c,2);
    x=configure(data,K,threshold);
    warmed=false;
    for i=1:numel(cfg.seeds)
        seed=cfg.seeds(i);
        pathScene=fullfile(cfg.folder,sprintf('condition%02d_scene%d.mat',c,seed));
        if isfile(pathScene)
            old=load(pathScene,'rows'); cells{i,c}=old.rows; continue
        end
        if ~warmed
            warmSeed=400000+c;
            assert(~ismember(warmSeed,cfg.seeds),'Reserved warm-up seed.');
            warm=channel_scene(x,warmSeed);
            % Warm only the fast LP path. Beampattern capture reports no time,
            % so an additional tutorial AO warm-up would repeat the slow solve.
            design_beams(warm,settings(struct('outer',1,'inner',1)),backend,model);
            warmed=true;
        end
        x=channel_scene(x,seed);
        jobs=make_jobs(cfg.budgets);
        orderStream=RandStream('mt19937ar','Seed',seed);
        records={}; captures={};
        for rep=1:cfg.repeats
            ids=1:numel(jobs);
            if rep>1, ids=ids(2:end); end % one AO score/time per scene
            ids=ids(randperm(orderStream,numel(ids)));
            for id=ids
                j=jobs(id);
                row=struct('scene',seed,'K',K,'gammaDB',threshold,'mode',j.mode, ...
                    'outer',j.outer,'inner',j.inner,'rep',rep,'seconds',NaN, ...
                    'trueDB',NaN,'designDB',NaN,'marginDB',NaN,'powerW',NaN, ...
                    'feasible',false,'steps',NaN,'status',"");
                watch=tic;
                try
                    if j.mode=="AO"
                        [W,U,row.seconds,row.steps]=run_ao(x,data,Vseed(seed));
                        row.status="completed";
                    else
                        opt=settings(j);
                        watch=tic;
                        [W,U,row.steps,row.status]=design_beams(x,opt,backend,model);
                        row.seconds=toc(watch);
                    end
                    [row.marginDB,row.powerW]=audit_beams(x,W);
                    row.feasible=row.marginDB>=-1e-4 && row.powerW<=x.P*(1+1e-7);
                    Q=pagemtimes(W,'none',W,'ctranspose');
                    row.trueDB=10*log10(compute_SCNR(Q,U,data.Vcc,x.R,x.sig,x.a,x.b));
                    row.designDB=10*log10(compute_SCNR(Q,U,x.V,x.R,x.sig,x.a,x.b));
                    assert(isfinite(row.trueDB) && isfinite(row.designDB),'Invalid score.');
                    captures{end+1,1}=struct('scene',seed,'K',K,'gammaDB',threshold, ...
                        'mode',j.mode,'outer',j.outer,'inner',j.inner,'rep',rep, ...
                        'W',W,'U',U); %#ok<AGROW>
                catch err
                    if isnan(row.seconds), row.seconds=toc(watch); end
                    row.feasible=false; row.status="error: "+string(err.message);
                end
                records{end+1,1}=row; %#ok<AGROW>
            end
        end
        rows=struct2table(vertcat(records{:}),'AsArray',true);
        tmp=[tempname(cfg.folder),'.mat']; save(tmp,'rows','captures'); movefile(tmp,pathScene,'f');
        cells{i,c}=rows;
        fprintf('K=%d, threshold=%g dB, scene %d (%d/%d), errors %d\n', ...
            K,threshold,seed,i,numel(cfg.seeds),nnz(startsWith(rows.status,"error:")));
    end
end
out.raw=vertcat(cells{:});
[out.scenes,out.summary,out.learning]=summarize(out.raw);
for name=["raw","scenes","summary","learning"]
    writetable(out.(name),fullfile(cfg.folder,name+".csv"));
end
out.folder=cfg.folder;
out.beampatternFiles=plot_beampatterns(cfg.folder,data);
disp(out.summary)
end

function x=channel_scene(x,seed)
rng(seed,'twister');
x.H=(randn(x.nt,x.N,x.K)+1i*randn(x.nt,x.N,x.K))/sqrt(2);
d=randi([50,400],1,1,x.K);
x.H=x.H.*10.^((-30-27*log10(d))/20);
end

function s=Vseed(seed)
s=mod(17389+1664525*double(seed),2^32);
end

function [W,U,seconds,it]=run_ao(x,data,seed)
p=data.para; p.N=x.N; p.L=64; p.Ku=x.K; p.Pt=x.P;
opt=struct('maxAO',30,'maxDinkel',30,'tolAO_dB',.01,'tolDinkel_eta',.001, ...
    'lambdaRidge',0,'verbose',0);
rng(seed,'twister'); gamma=repmat(x.gamma,x.N,x.K);
timer=tic;
v=solve_BLP_AO(p,x.sig,x.V,x.R,x.a,x.b,x.H,gamma,x.noise,opt);
seconds=toc(timer); W=cat(2,v.w_comm,v.Ws); U=v.u; it=numel(v.scnr_hist)-1;
end

function jobs=make_jobs(budgets)
jobs=struct('mode',"AO",'outer',0,'inner',0);
for b=1:size(budgets,1)
    for m=["physics","fixedProposal","learnedProposal"]
        jobs(end+1)=struct('mode',m,'outer',budgets(b,1),'inner',budgets(b,2)); %#ok<AGROW>
    end
end
end

function [S,A,L]=summarize(T)
key={'K','gammaDB','scene','mode','outer','inner'};
S=unique(T(:,key),'rows'); S.seconds=NaN(height(S),1); S.trueDB=S.seconds;
S.feasible=false(height(S),1); S.aoValid=S.feasible; S.hit=S.feasible;
S.gapDB=S.seconds; S.speedup=S.seconds;
for i=1:height(S)
    q=S(i,:); same=T.K==q.K & T.gammaDB==q.gammaDB & T.scene==q.scene;
    r=T(same & T.mode==q.mode & T.outer==q.outer & T.inner==q.inner,:);
    a=T(same & T.mode=="AO",:);
    S.seconds(i)=median(r.seconds); S.feasible(i)=all(r.feasible); S.aoValid(i)=a.feasible;
    if S.feasible(i), S.trueDB(i)=median(r.trueDB); end
    if a.feasible && S.feasible(i)
        S.gapDB(i)=a.trueDB-S.trueDB(i); S.hit(i)=S.gapDB(i)<=0.1;
        S.speedup(i)=a.seconds/S.seconds(i);
    end
end
A=unique(S(:,{'K','gammaDB','mode','outer','inner'}),'rows');
for i=1:height(A)
    q=A(i,:); r=S(S.K==q.K & S.gammaDB==q.gammaDB & S.mode==q.mode ...
        & S.outer==q.outer & S.inner==q.inner,:);
    A.attempted(i)=height(r); A.feasible(i)=nnz(r.feasible); A.validAO(i)=nnz(r.aoValid);
    A.hits(i)=nnz(r.hit); A.medianSeconds(i)=median(r.seconds);
    A.medianGapDB(i)=median(r.gapDB,'omitnan'); A.pairedSpeedup(i)=median(r.speedup,'omitnan');
    A.hitSpeedup(i)=median(r.speedup(r.hit),'omitnan');
end
F=S(S.mode=="fixedProposal",:); L=S(S.mode=="learnedProposal",:);
key={'K','gammaDB','scene','outer','inner'};
F=sortrows(F,key); L=sortrows(L,key); assert(isequal(F(:,key),L(:,key)));
L.fixedDB=F.trueDB; L.learnedMinusFixedDB=L.trueDB-F.trueDB;
L.fixedSeconds=F.seconds; L.fixedHit=F.hit; L.learnedOnlyHit=L.hit & ~F.hit;
L.fixedOnlyHit=F.hit & ~L.hit;
end

function o = settings(given)
o = struct('outer',5,'inner',3,'toleranceDB',0.002,'mode',"physics");
keys = fieldnames(given);
assert(all(ismember(keys,fieldnames(o))),'Unknown option.');
for j = 1:numel(keys), o.(keys{j}) = given.(keys{j}); end
for field = ["outer","inner"]
    validateattributes(o.(field),{'numeric'},{'integer','scalar','positive','finite'});
end
validateattributes(o.toleranceDB,{'numeric'},{'real','scalar','nonnegative','finite'});
o.mode = string(o.mode);
assert(isscalar(o.mode) && any(o.mode == ["physics","learnedProposal","fixedProposal"]), ...
    'Invalid proposal mode.');
end

function x = configure(data,K,gammaDB)
p = data.para;
x.N = 32; x.P = 100; x.nt = p.Nt; x.nr = p.Nr; x.ns = p.Ns; x.K = K;
assert(K<x.ns && K<x.nt,'The feasible seed needs spare transmit dimensions and a sensing stream.');
x.noise = 1e-19*p.deltaf;
x.gamma = 10^(gammaDB/10);
x.V = data.Vcc_hat(1:x.N);
x.R = data.R_eta(:,:,1:x.N);
f = p.fc + (0:x.N-1)*p.deltaf;
x.sig = x.P*(p.c0./f).^2*1e4*10^.5/((4*pi)^3*41.8^4);
for n = 1:x.N
    phase = -1i*pi*sind(-10)*f(n)/p.fc;
    x.a{n,1} = exp((0:x.nt-1)'*phase)/sqrt(x.nt);
    x.b{n,1} = exp((0:x.nr-1)'*phase)/sqrt(x.nr);
end
end

function name = choose_backend()
% Probe the actual calling convention and inequality-price sign. Some
% installations expose linprog on the path but reject its options or omit
% dual outputs. This probe runs once, outside the reported design time.
reasons = strings(0,1);
for name = ["linprog","linprog-default","CVX"]
    try
        [z,cost,price] = solve_lp(name,[-1;0],sparse([1,-1]),sparse([1,1]));
        assert(norm(z-[0.5;0.5],inf)<1e-5 && abs(cost+0.5)<1e-5 ...
            && isscalar(price) && abs(price-0.5)<1e-5, ...
            'Unexpected primal solution or inequality multiplier.');
        return;
    catch err
        reasons(end+1,1) = name + ": " + string(err.message); %#ok<AGROW>
    end
end
error('No LP backend passed the startup test.\n%s',char(join(reasons,newline)));
end

function [W,U,steps,status] = design_beams(x,o,backend,model)
% Alternate receive filtering and transmit design. Each LP allocates power
% jointly over ALL tones. Dual multipliers identify new useful directions.
[bank,G,noiseRatio,W] = initial_bank(x);
[eta,A,C,hot,cold,U] = receiver(x,W);
steps = 0; status = "budget reached";
for j = 1:o.outer
    entryScore = eta;
    bestFixed = eta;
    if o.mode ~= "physics"
        bank = add_policy_beams(bank,x,W,A,C,eta,hot,cold,j,o.outer,model,o.mode);
    end
    for k = 1:o.inner
        [powers,dual,values] = allocate_power(x,bank,G,noiseRatio,A,C,sum(hot),backend);
        steps = steps+1;
        Y = recover_streams(x,G,bank,powers);
        [margin,power] = audit_beams(x,Y);
        if margin >= -1e-4 && power <= x.P*(1+1e-7)
            fixedScore = fixed_ratio(Y,A,C,sum(hot));
            if fixedScore >= bestFixed*(1-1e-10)
                W = Y;
                bestFixed = fixedScore;
            end
        end
        [bank,boundDB] = enrich_directions(x,bank,G,dual,values);
        if boundDB <= o.toleranceDB, break; end
    end
    [eta,A,C,hot,cold,U] = receiver(x,W);
    if 10*log10(eta/entryScore) < o.toleranceDB && boundDB <= o.toleranceDB
        status = "tolerance reached"; break;
    end
end
end

function [bank,G,noiseRatio,W] = initial_bank(x)
% Owner 0 denotes sensing. Owners 1:K denote communication users.
bank = cell(x.N,1); G = cell(x.N,1); noiseRatio = zeros(x.K,x.N);
W = zeros(x.nt,x.ns,x.N);
for n = 1:x.N
    H = reshape(x.H(:,n,:),x.nt,x.K);
    norms = vecnorm(H);
    G{n} = H./norms;
    noiseRatio(:,n) = (x.noise./(x.P*norms.^2)).';
    Z = pinv(G{n}');                % Unit desired response and zero cross-talk
    a = x.a{n};
    responses = G{n}'*a;
    free = a - Z*responses;          % Sensing in the users' null space
    if norm(free) < 1e-10
        basis = null(G{n}'); free = basis(:,1);
    end
    V = [Z./vecnorm(Z), free/norm(free), a];
    owner = [1:x.K, 0, 0];
    for k = 1:x.K
        v = free + Z(:,k)*responses(k);
        V = [V, a, v/norm(v)]; %#ok<AGROW>
        owner = [owner, k, k]; %#ok<AGROW>
    end
    bank{n} = struct('V',V,'owner',owner);
    W(:,1:x.K,n) = Z.*sqrt(x.gamma*(1+2e-5)*x.P*noiseRatio(:,n).');
    W(:,x.K+1,n) = free/norm(free);
end
used = sum(abs(W(:,1:x.K,:)).^2,'all');
assert(used<x.P,'The zero-forcing initialization exceeds the power budget.');
W(:,x.K+1,:) = W(:,x.K+1,:).*sqrt((x.P-used)/x.N);
end

function [eta,A,C,hot,cold,U] = receiver(x,W)
Q = pagemtimes(W,'none',W,'ctranspose');
U = zeros(x.nr,x.N); A = zeros(x.nt,x.nt,x.N); C = A;
hot = zeros(1,x.N); cold = hot;
for n = 1:x.N
    clutter = kernel_apply(x.V{n},Q(:,:,n),x.nr);
    u = mvdr_beamformer(clutter+x.R(:,:,n),x.b{n},0);
    U(:,n) = u;
    A(:,:,n) = x.sig(n)*abs(u'*x.b{n})^2*(x.a{n}*x.a{n}');
    C(:,:,n) = build_Ac(x.V{n},u,x.nt);
    hot(n) = real(u'*x.R(:,:,n)*u);
    cold(n) = real(u'*clutter*u);
end
eta = fixed_ratio(W,A,C,sum(hot));
end

function eta = fixed_ratio(W,A,C,hot)
num = real(sum(conj(W).*pagemtimes(A,W),'all'));
den = hot + real(sum(conj(W).*pagemtimes(C,W),'all'));
eta = num/den;
end

function [power,dual,v] = allocate_power(x,bank,G,noiseRatio,A,C,hot,backend)
% For each candidate (unit) direction, power contributes linearly to
% target return, clutter, and user signals. Change variables z=t*p/Psafe
% and normalize t+clutter'*z=1 to solve the fractional objective as one LP.
safePower = x.P*(1-1e-8);
gamma = x.gamma*(1+2e-5);
noiseRatio = noiseRatio*x.P/safePower;
rowScale = 1./sqrt(noiseRatio);
blocks = cell(x.N,1); reward = cell(x.N,1); penalty = cell(x.N,1);
width = zeros(x.N,1);
for n = 1:x.N
    V = bank{n}.V;
    reward{n} = (safePower/hot)*real(sum(conj(V).*(A(:,:,n)*V),1));
    penalty{n} = (safePower/hot)*max(0,real(sum(conj(V).*(C(:,:,n)*V),1)));
    desired = ((1:x.K).' == bank{n}.owner);
    coeff = (1+1/gamma)*desired-1;
    blocks{n} = sparse(-rowScale(:,n).*coeff.*abs(G{n}'*V).^2);
    width(n) = size(V,2);
end
a = [reward{:}]; c = [penalty{:}]; scale = max(a);
assert(scale>0 && hot>0,'Degenerate sensing objective.');
B = blkdiag(blocks{:});
scaledNoise = noiseRatio.*rowScale;
ineq = [B, sparse(scaledNoise(:)); sparse(ones(1,sum(width))), -1];
eq = sparse([c,1]); objective = -[a/scale,0].';
[solution,cost,mult] = solve_lp(backend,objective,ineq,eq);
t = solution(end);
assert(isfinite(t) && t>0,'Invalid fractional power normalization.');
power = mat2cell(max(0,solution(1:end-1))*safePower/t,width,1);
dual.user = reshape(mult(1:end-1),x.K,x.N).*rowScale;
dual.power = mult(end);
v.ratio = -cost; v.gamma = gamma;
v.matrix = (safePower/hot).*(A./scale - v.ratio.*C);
end

function [z,cost,mult] = solve_lp(backend,f,ineq,eq)
rhs = zeros(size(ineq,1),1);
if backend ~= "CVX"
    if backend == "linprog"
        opts = optimoptions('linprog','Display','none','ConstraintTolerance',1e-9, ...
            'OptimalityTolerance',1e-9);
        [z,cost,flag,msg,lambda] = linprog(f,ineq,rhs,eq,1,zeros(size(f)),[],opts);
    else
        [z,cost,flag,msg,lambda] = linprog(f,ineq,rhs,eq,1,zeros(size(f)),[]);
    end
    assert(flag>0,'Power LP failed: %s',msg.message);
    mult = lambda.ineqlin;
else
    m = numel(f);
    cvx_begin quiet
        variable z(m) nonnegative
        dual variable price
        minimize(f.'*z)
        subject to
            price: ineq*z <= rhs;
            eq*z == 1;
    cvx_end
    assert(contains(cvx_status,'Solved'),'CVX power LP failed: %s',cvx_status);
    cost = cvx_optval; mult = price;
end
end

function [bank,boundDB] = enrich_directions(x,bank,G,dual,v)
% Leading eigenvector maximizes reduced cost over all unit directions
% for one tone and stream owner. Add it when the current LP can improve.
excess = 0;
for n = 1:x.N
    H = G{n}; mu = dual.user(:,n);
    base = v.matrix(:,:,n) - (H.*mu.')*H';
    for owner = [1:x.K, 0]
        M = base;
        if owner>0
            h = H(:,owner);
            M = M + (1+1/v.gamma)*mu(owner)*(h*h');
        end
        [modes,ev] = eig((M+M')/2,'vector');
        [largest,idx] = max(real(ev));
        improvement = largest-dual.power;
        excess = max(excess,improvement);
        if improvement>1e-7
            bank{n}.V(:,end+1) = modes(:,idx);
            bank{n}.owner(end+1) = owner;
        end
    end
end
% Reduced-cost bound for the tightened, fixed-receiver allocation problem.
boundDB = 10*log10(1+excess/max(v.ratio,eps));
end

function W = recover_streams(x,G,bank,power)
% Compress each user's positive-semidefinite covariance to one beam
% preserving its desired power. Use the remaining covariance for sensing.
W = zeros(x.nt,x.ns,x.N);
for n = 1:x.N
    V = bank{n}.V.*sqrt(power{n}.');
    for k = 1:x.K
        Vk = V(:,bank{n}.owner==k);
        response = Vk'*G{n}(:,k);
        if norm(response)>0, W(:,k,n) = Vk*(response/norm(response)); end
    end
    comm = W(:,1:x.K,n);
    residual = V*V'-comm*comm';
    [modes,ev] = eig((residual+residual')/2,'vector');
    [ev,order] = sort(real(ev),'descend');
    keep = 1:(x.ns-x.K);
    W(:,x.K+1:end,n) = modes(:,order(keep)).*sqrt(max(0,ev(keep).'));
end
end

function [margin,power] = audit_beams(x,W)
minimum = inf;
for n = 1:x.N
    H = reshape(x.H(:,n,:),x.nt,x.K);
    gains = abs(H'*W(:,:,n)).^2;
    indices = sub2ind(size(gains),1:x.K,1:x.K);
    signal = gains(indices).';
    gains(indices) = 0;
    minimum = min(minimum,min(signal./(x.noise+sum(gains,2))));
end
margin = 10*log10(minimum/x.gamma);
power = sum(abs(W).^2,'all');
end

function bank = add_policy_beams(bank,x,W,A,C,eta,hot,cold,j,depth,model,mode)
% Optional ablation. Reuse the trained transmit block from run_tutorial_WU.
% The LP can assign zero power to any proposed direction.
w = model.W(:);
if mode=="fixedProposal", w(:)=0; end
for n = 1:x.N
    H = reshape(x.H(:,n,:),x.nt,x.K); Y = W(:,:,n);
    response = H'*Y; desired = diag(response(:,1:x.K));
    response(sub2ind(size(response),1:x.K,1:x.K)) = 0;
    interference = sum(abs(response).^2,2)+x.noise;
    ratio = abs(desired).^2./(interference*x.gamma);
    qos = zeros(size(Y));
    for k = 1:x.K
        grad = -H(:,k)*response(k,:)/interference(k);
        grad(:,k) = H(:,k)*desired(k)/max(abs(desired(k))^2,realmin);
        qos = qos + grad/(max(norm(grad,'fro'),realmin)*max(1,ratio(k))^2);
    end
    f = [cold(n)/max(hot(n)+cold(n),realmin); tanh(log(max(min(ratio),realmin))); ...
         tanh(log(max(norm(Y,'fro')^2*x.N/x.P,realmin))); j/depth];
    hidden = tanh(reshape(w(1:24),6,4)*f+w(25:30));
    logits = tanh(reshape(w(31:54),4,6)*hidden+w(55:58));
    target = A(:,:,n)*Y; clutter = eta*C(:,:,n)*Y;
    direction = (exp(logits(1))*target-exp(logits(2))*clutter) ...
        /max(norm(target,'fro')+norm(clutter,'fro'),realmin) ...
        +0.15*(1+logits(3))*qos/x.K;
    proposed = Y+sqrt(x.P/x.N)*model.fixed*exp(logits(4))*direction;
    V = proposed(:,1:x.K);
    valid = vecnorm(V)>0;
    bank{n}.V = [bank{n}.V, V(:,valid)./vecnorm(V(:,valid))];
    bank{n}.owner = [bank{n}.owner, find(valid)];
end
end

function outputFiles = plot_beampatterns(folder,data)
% Plot the total transmit covariance across all OFDM tones. Curves share one
% normalization because every method uses the same 100 W power constraint.
files = dir(fullfile(folder,'condition*_scene*.mat'));
outputFiles = strings(0,1);
scan = (-90:0.25:90).';
methodOrder = ["AO","physics","fixedProposal","learnedProposal"];
methodLabels = ["AO","Physics pricing","Fixed proposal","Learned proposal"];
styles = {'k--','-o','-s','-^'};
colors = [0.12 0.12 0.12; 0.12 0.35 0.60; 0.75 0.34 0.10; 0.05 0.48 0.38];
targetAngles = data.tarPara.theta(:);
dominantClutter = dominant_clutter_angles(data.theta_cc_all, ...
    data.sigma2_cc_all,3,8);

for f = 1:numel(files)
    saved = load(fullfile(files(f).folder,files(f).name),'captures');
    if ~isfield(saved,'captures') || isempty(saved.captures)
        continue
    end
    captures = [saved.captures{:}];
    names = string({captures.mode});
    curves = nan(numel(scan),numel(methodOrder));
    used = false(1,numel(methodOrder));
    for m = 1:numel(methodOrder)
        pick = find(names == methodOrder(m),1);
        if isempty(pick), continue; end
        curves(:,m) = transmit_pattern(captures(pick).W,scan,data.para);
        used(m) = true;
    end
    assert(all(used),'AO or one accelerated beamformer is missing from the capture.');
    scale = max(curves(:,used),[],'all');
    curvesDB = 10*log10(max(curves/scale,realmin));

    first = captures(find(used,1));
    fig = figure('Color','w','Visible','off','Position',[100 100 920 470]);
    closeFigure = onCleanup(@() close(fig)); %#ok<NASGU>
    ax = axes('Parent',fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    handles = gobjects(0,1); labels = strings(0,1);
    for m = find(used)
        style = styles{m};
        handles(end+1) = plot(ax,scan,curvesDB(:,m),style, ...
            'Color',colors(m,:),'LineWidth',1.45,'MarkerIndices',1:48:numel(scan), ...
            'MarkerSize',4); %#ok<AGROW>
        labels(end+1) = methodLabels(m); %#ok<AGROW>
    end
    for angle = targetAngles.'
        xline(ax,angle,'-.','Target','Color',[0.12 0.35 0.75], ...
            'LineWidth',1.1,'LabelVerticalAlignment','bottom');
    end
    for angle = dominantClutter.'
        xline(ax,angle,':','Color',[0.72 0.18 0.18],'LineWidth',1.0, ...
            'HandleVisibility','off');
    end
    xlabel(ax,'Azimuth angle (degrees)');
    ylabel(ax,'Normalized transmit power (dB)');
    title(ax,sprintf('%d users, %g dB SINR, seed %d, 15 allocations', ...
        first.K,first.gammaDB,first.scene));
    xlim(ax,[-90 90]); ylim(ax,[-35 1]);
    legend(ax,handles,labels,'Location','southoutside','Orientation','horizontal');
    text(ax,0.99,0.03,'Red dotted lines: three dominant cold-clutter directions', ...
        'Units','normalized','HorizontalAlignment','right','FontSize',8);

    stem = sprintf('beampattern_K%d_G%g_seed%d',first.K,first.gammaDB,first.scene);
    pdfFile = fullfile(folder,[stem '.pdf']);
    pngFile = fullfile(folder,[stem '.png']);
    exportgraphics(fig,pdfFile,'ContentType','vector');
    exportgraphics(fig,pngFile,'Resolution',240);
    outputFiles(end+1:end+2,1) = [string(pdfFile);string(pngFile)]; %#ok<AGROW>

    summary = pattern_summary(methodOrder(used),curvesDB(:,used),scan, ...
        targetAngles,dominantClutter);
    writetable(summary,fullfile(folder,[stem '_summary.csv']));
end
end

function power = transmit_pattern(W,angles,para)
tones = size(W,3); antennas = size(W,1);
frequencies = para.fc+(0:tones-1)*para.deltaf;
power = zeros(numel(angles),1);
indices = (0:antennas-1).';
for n = 1:tones
    steering = exp(-1i*pi*indices*sind(angles.')*frequencies(n)/para.fc) ...
        /sqrt(antennas);
    responses = steering'*W(:,:,n);
    power = power+sum(abs(responses).^2,2);
end
end

function chosen = dominant_clutter_angles(angles,powers,count,separation)
angles = angles(:); weights = mean(real(powers),2);
[~,order] = sort(weights,'descend');
chosen = zeros(0,1);
for index = order.'
    if isempty(chosen) || all(abs(angles(index)-chosen)>=separation)
        chosen(end+1,1) = angles(index); %#ok<AGROW>
        if numel(chosen)==count, break; end
    end
end
chosen = sort(chosen);
end

function summary = pattern_summary(methods,curvesDB,scan,targetAngles,clutterAngles)
count = numel(methods);
peakAngle = nan(count,1); targetPowerDB = nan(count,1);
strongestClutterDB = nan(count,1); targetToClutterDB = nan(count,1);
targetIndex = arrayfun(@(x) find(abs(scan-x)==min(abs(scan-x)),1),targetAngles);
clutterIndex = arrayfun(@(x) find(abs(scan-x)==min(abs(scan-x)),1),clutterAngles);
for m = 1:count
    [~,peak] = max(curvesDB(:,m)); peakAngle(m) = scan(peak);
    targetPowerDB(m) = max(curvesDB(targetIndex,m));
    strongestClutterDB(m) = max(curvesDB(clutterIndex,m));
    targetToClutterDB(m) = targetPowerDB(m)-strongestClutterDB(m);
end
summary = table(methods(:),peakAngle,targetPowerDB,strongestClutterDB, ...
    targetToClutterDB,'VariableNames',{'method','peakAngle','targetPowerDB', ...
    'strongestDominantClutterDB','targetToDominantClutterDB'});
end
