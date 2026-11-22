function [fig,summary,folder] = run_sinr_three_methods(repo,outWU)
% Compare original AO, fixed and trained proposals on matched 50 dBm channels.
% [fig,summary,folder] = run_sinr_three_methods(repo,outWU)
% outWU may instead be the full path to a trained_model.mat checkpoint.
if isstruct(outWU)
    model=outWU.model;
else
    ck=load(outWU,'model'); model=ck.model;
end
assert(isfield(model,'W') && numel(model.W)==58 && all(isfinite(model.W(:))), ...
    'Expected the trained 4-6-4 W network from run_tutorial_WU.');
validateattributes(model.fixed,{'numeric'},{'positive','finite','scalar'});
fn=fullfile(repo,'function'); datafile=fullfile(repo,'data','covMat_BLP.mat');
assert(isfile(datafile) && isfolder(fn),'Pass the root of the tutorial repo.');
oldPath=path; undoPath=onCleanup(@() path(oldPath)); %#ok<NASGU>
oldRng=rng; undoRng=onCleanup(@() rng(oldRng)); %#ok<NASGU>
addpath(fn);
seeds=11001:11008; users=[4 8 12]; thresholds=0:4:20;
folder=fullfile(repo,'ao_fixed_learned_sinr_50dBm');
if ~isfolder(folder), mkdir(folder); end
assert(isfile(fullfile(fn,'solve_BLP_AO.m')),'Missing authors'' AO solver.');
backend=choose_backend();
d=load(datafile,'para','Vcc','Vcc_hat','R_eta');
meta=dir(datafile);
signature=struct('W',model.W,'step',model.fixed,'seeds',seeds,'users',users, ...
    'thresholds',thresholds,'backend',backend,'dataRoot',char(repo), ...
    'aoSource',fileread(fullfile(fn,'solve_BLP_AO.m')),'dataSize',meta.bytes,'dataTime',meta.datenum,'driver',fileread([mfilename('fullpath') '.m']));
manifest=fullfile(folder,'settings.mat');
if isfile(manifest)
    old=load(manifest,'signature');
    assert(isequaln(signature,old.signature), ...
        'Model or settings changed. Move the existing output folder before starting a new sweep.');
else
    save(manifest,'signature');
end
results=cell(numel(users),numel(thresholds),numel(seeds));
for k=1:numel(users)
    K=users(k);
    for g=1:numel(thresholds)
        gamma=thresholds(g);
        x=configure(d,K,gamma); % same 100 W (50 dBm) setting as the previous SINR sweep
        truth=d.Vcc(1:x.N); estimate=d.Vcc_hat(1:x.N);
        for i=1:numel(seeds)
            seed=seeds(i);
            checkpoint=fullfile(folder,sprintf('K%d_g%d_s%d.mat',K,gamma,seed));
            if isfile(checkpoint)
                saved=load(checkpoint,'rows'); results{k,g,i}=saved.rows; continue
            end
            rng(seed,'twister');
            x.H=(randn(x.nt,x.N,K)+1i*randn(x.nt,x.N,K))/sqrt(2);
            distance=randi([50 400],1,1,K);
            x.H=x.H.*10.^((-30-27*log10(distance))/20);
            base=struct('study',"sinr",'dbm',50,'K',K,'gamma',gamma,'seed',seed, ...
                'method',"",'kernel',"",'trueDB',NaN,'designDB',NaN,'marginDB',NaN, ...
                'powerW',NaN,'lpSolves',NaN,'status',"ok");
            rows=repmat(base,6,1); n=0;
            for kernel=["perfect","estimated"]
                if kernel=="perfect", x.V=truth; else, x.V=estimate; end
                for method=["AO","fixedProposal","learnedProposal"]
                    n=n+1; row=base; row.kernel=kernel; row.method=method;
                    try
                        if method=="AO"
                            p=d.para; p.N=x.N; p.L=64; p.Ku=K; p.Pt=x.P;
                            opt=struct('maxAO',30,'maxDinkel',30,'tolAO_dB',.01, ...
                                'tolDinkel_eta',.001,'lambdaRidge',0,'verbose',0);
                            rng(mod(17389+1664525*double(seed),2^32),'twister');
                            a=solve_BLP_AO(p,x.sig,x.V,x.R,x.a,x.b,x.H, ...
                                repmat(x.gamma,x.N,K),x.noise,opt);
                            W=cat(2,a.w_comm,a.Ws); U=a.u;
                        else
                            opt=settings(struct('mode',method,'outer',3,'inner',2));
                            [W,U,row.lpSolves]=design_beams(x,opt,backend,model,false,truth);
                        end
                        [row.marginDB,row.powerW]=audit_beams(x,W);
                        Q=pagemtimes(W,'none',W,'ctranspose');
                        row.trueDB=10*log10(compute_SCNR(Q,U,truth,x.R,x.sig,x.a,x.b));
                        row.designDB=10*log10(compute_SCNR(Q,U,x.V,x.R,x.sig,x.a,x.b));
                        assert(isfinite(row.trueDB) && isfinite(row.designDB),'Invalid score.');
                        if ~(row.marginDB>=-1e-4 && row.powerW<=x.P*(1+1e-7))
                            row.status="infeasible";
                        end
                    catch err
                        if method=="AO" && (contains(lower(err.message),'missing solver') || ...
                                contains(lower(err.message),'unknown, unusable') || ...
                                contains(lower(err.message),'undefined function') || contains(lower(err.message),'license'))
                            rethrow(err);
                        end
                        row.status="error: "+string(err.message);
                        warning('Sweep:Unsuccessful','K=%d SINR=%g seed=%d %s %s: %s', ...
                            K,gamma,seed,char(kernel),char(method),err.message);
                    end
                    rows(n)=row;
                end
            end
            results{k,g,i}=rows;
            tmp=[tempname(folder) '.mat']; save(tmp,'rows'); movefile(tmp,checkpoint,'f');
        end
        fprintf('Completed K=%d, SINR threshold=%g dB.\n',K,gamma);
    end
end
T=struct2table(vertcat(results{:}));
assert(height(T)==864,'Expected three methods and two kernels at every channel draw.');
writetable(T,fullfile(folder,'sweep_trials.csv'));
[fig,summary]=plot_sinr_three_methods(folder);
end

function row=observe(x,W,U,eta,truth,it,steps)
Q=pagemtimes(W,'none',W,'ctranspose');
[margin,power]=audit_beams(x,W);
trueDB=10*log10(compute_SCNR(Q,U,truth,x.R,x.sig,x.a,x.b));
row=[it,steps,10*log10(eta),trueDB,margin,power];
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

function [W,U,steps,status,trajectory] = design_beams(x,o,backend,model,record,truth)
% Alternate receive filtering and transmit design. Each LP allocates power
% jointly over ALL tones. Dual multipliers identify new useful directions.
[bank,G,noiseRatio,W] = initial_bank(x);
[eta,A,C,hot,cold,U] = receiver(x,W);
steps = 0; status = "budget reached";
trajectory = zeros(0,6);
if record, trajectory = observe(x,W,U,eta,truth,0,0); end
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
    if record, trajectory(end+1,:) = observe(x,W,U,eta,truth,j,steps); end %#ok<AGROW>
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
