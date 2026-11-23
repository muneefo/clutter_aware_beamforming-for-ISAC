function out = fast_tutorial_beams(repo, outWU, options, data)
% FAST_TUTORIAL_BEAMS   Fast transmit design for the tutorial's block precoder.
%   out = fast_tutorial_beams(repo, outWU)
%   out = fast_tutorial_beams(repo, outWU, struct('mode',"learnedProposal"))
%
% The default mode ("physics") was benchmarked in Python. "fixedProposal"
% and "learnedProposal" additionally propose one beam per user at each outer
% iteration, using zero or saved W-network weights. All three use exact MVDR.
% Requires the tutorial folder and linprog (or CVX as a slower LP backend).
% The saved AO measurements and true kernel are used only for evaluation.
if nargin < 3, options = struct(); end
options = settings(options);
oldPath = path; cleanPath = onCleanup(@() path(oldPath)); %#ok<NASGU>
oldRng = rng; cleanRng = onCleanup(@() rng(oldRng)); %#ok<NASGU>
addpath(fullfile(repo, 'function'));
if nargin < 4
    data = load(fullfile(repo,'data','covMat_BLP.mat'), 'para','Vcc_hat','Vcc','R_eta');
end
cfg = outWU.options;
x = configure(data, cfg.K, cfg.gammaDB);
ao = outWU.trials(outWU.trials.method == "AO_SDP", :);
assert(~isempty(ao) && numel(unique(ao.scene)) == height(ao), 'Expected one AO row per scene.');
backend = choose_backend();
if options.ablation ~= "" || options.matchedPolicy ~= ""
    assert(backend == "linprog", 'Matched experiments require explicit-tolerance linprog.');
end
model = [];
if options.mode ~= "physics" || any(options.matchedPolicy == ["reducedCost","envelope"])
    model = outWU.model;
end
rows = cell(height(ao),1); designs = cell(height(ao),1);
profiles = cell(height(ao),1); plans = cell(height(ao),1);
if ~options.quiet, fprintf('Mode %s, LP backend %s.\n', options.mode, backend); end
for i = 1:height(ao)
    seed = ao.scene(i);
    rng(seed,'twister');
    H = (randn(x.nt,x.N,x.K) + 1i*randn(x.nt,x.N,x.K))/sqrt(2);
    distance = randi([50,400],1,1,x.K);
    x.H = H .* 10.^((-30-27*log10(distance))/20);
    if options.ablation == "B7"
        plan = options.pricingSchedules{i};
    else
        plan = {};
    end
    timer = tic;
    [W,U,steps,status,profiles{i},plans{i}] = design_beams(x,options,backend,model,seed,plan);
    seconds = toc(timer);
    [margin,power] = audit_beams(x,W);
    feasible = margin >= -1e-4 && power <= x.P*(1+1e-7);
    Q = pagemtimes(W,'none',W,'ctranspose');
    trueDB = 10*log10(compute_SCNR(Q,U,data.Vcc,x.R,x.sig,x.a,x.b));
    gap = NaN;
    if feasible && ao.feasible(i), gap = ao.trueSCNRdB(i) - trueDB; end
    rows{i} = struct('scene',seed,'mode',options.mode,'trueSCNRdB',trueDB, ...
        'aoSCNRdB',ao.trueSCNRdB(i),'gapToAOdB',gap,'seconds',seconds, ...
        'aoSeconds',ao.seconds(i),'minMarginDB',margin,'powerW',power, ...
        'feasible',feasible,'hitQuality',gap<=cfg.qualityGapDB, ...
        'lpSolves',steps,'status',status);
    designs{i} = struct('W',W,'U',U);
    if options.matchedPolicy ~= ""
        rows{i}.matchedPolicy = options.matchedPolicy;
        rows{i}.designSCNRdB = 10*log10(compute_SCNR(Q,U,x.V,x.R,x.sig,x.a,x.b));
    end
    if ~options.quiet
        fprintf('Scene %d: %.4f dB, AO gap %+.4f dB, %.3f s, %s\n', ...
            seed,trueDB,gap,seconds,status);
    end
end
out = struct('trials',struct2table(vertcat(rows{:}),'AsArray',true), ...
        'designs',{designs},'options',options,'backend',backend);
if options.ablation ~= "" || options.matchedPolicy ~= ""
    out.timing = profiles;
    out.pricingPlans = plans;
end
if ~options.quiet, disp(out.trials); end
end

function o = settings(given)
o = struct('outer',5,'inner',3,'toleranceDB',0.002,'mode',"physics", ...
    'ablation',"",'pricingSchedules',{{}},'matchedPolicy',"",'quiet',false);
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
o.ablation = string(o.ablation);
assert(isscalar(o.ablation) && any(o.ablation == ["","B6","B7","B8"]), 'Invalid Part B arm.');
assert(o.ablation == "" || o.mode == "physics", 'Part B uses the physics-only proposal.');
o.matchedPolicy = string(o.matchedPolicy);
assert(isscalar(o.matchedPolicy) && any(o.matchedPolicy == ["","analytic","reducedCost","envelope"]), ...
    'Invalid matched policy.');
assert(o.matchedPolicy == "" || (o.ablation == "" && o.mode == "physics"), ...
    'Use the matched Part E schedule on its own.');
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

function [W,U,steps,status,timing,plan] = design_beams(x,o,backend,model,seed,matched)
% Allocate across all tones. Part B enables exclusive stage clocks and matched budgets.
partE = o.matchedPolicy ~= "";
track = o.ablation ~= "" || partE;
timing = struct('build',0,'solve',0,'price',0,'mvdr',0,'kernel',0,'recovery',0);
plan = {};
if track, watch = tic; end
[bank,G,noiseRatio,W] = initial_bank(x);
if track, timing.build = toc(watch); end
if partE
    timing.inference = 0;
    timing.appendedAtoms = 0;
    timing.firstUserProposals = 0;
    firstWidth = size(bank{1}.V,2);
    everActive = false(x.K,x.N);
end
if o.ablation == "B7", random = RandStream('mt19937ar','Seed',370000+seed); end
[eta,A,C,hot,cold,U,timing] = receiver(x,W,track,timing);
steps = 0; status = "budget reached";
for j = 1:o.outer
    entryScore = eta;
    bestFixed = eta;
    if o.mode ~= "physics"
        bank = add_policy_beams(bank,x,W,A,C,eta,hot,cold,j,o.outer,model,o.mode);
    end
    for k = 1:o.inner
        [powers,dual,values,timing] = allocate_power(x,bank,G,noiseRatio,A,C,sum(hot),backend,track,timing);
        steps = steps+1;
        if partE && timing.firstUserProposals > 0
            for n = 1:x.N
                firstPowers = powers{n}(firstWidth+(1:x.K));
                everActive(:,n) = everActive(:,n) | (firstPowers(:) > 1e-8*x.P*(1-1e-8));
            end
        end
        if track, watch = tic; end
        Y = recover_streams(x,G,bank,powers);
        [margin,power] = audit_beams(x,Y);
        if margin >= -1e-4 && power <= x.P*(1+1e-7)
            fixedScore = fixed_ratio(Y,A,C,sum(hot));
            if fixedScore >= bestFixed*(1-1e-10)
                W = Y;
                bestFixed = fixedScore;
            end
        end
        if track, timing.recovery = timing.recovery + toc(watch); end
        terminal = j == o.outer && k == o.inner;
        if partE
            if steps == 1 && ~terminal && o.matchedPolicy ~= "analytic"
                [bank,inference,pricing] = append_l2o_directions(x,bank,G,noiseRatio,dual,values,model);
                timing.inference = timing.inference + inference;
                timing.price = timing.price + pricing;
            else
                watch = tic;
                [bank,boundDB] = enrich_directions(x,bank,G,dual,values,~terminal,true);
                timing.price = timing.price + toc(watch);
            end
            if ~terminal
                timing.appendedAtoms = timing.appendedAtoms + x.N*(x.K+1);
                if steps == 1, timing.firstUserProposals = x.N*x.K; end
            else
                timing.terminalBoundDB = boundDB;
            end
        elseif o.ablation == "B7" && ~terminal
            watch = tic;
            tags = matched{steps};
            for n = 1:x.N
                count = numel(tags{n});
                if count == 0, continue; end
                V = randn(random,x.nt,count) + 1i*randn(random,x.nt,count);
                bank{n}.V = [bank{n}.V, V./vecnorm(V)];
                bank{n}.owner = [bank{n}.owner, tags{n}];
            end
            timing.build = timing.build + toc(watch);
            plan{steps} = tags; %#ok<AGROW>
        elseif o.ablation == "B8" || ~track
            if track, watch = tic; end
            [bank,boundDB,tags] = enrich_directions(x,bank,G,dual,values,~track || ~terminal);
            if track
                timing.price = timing.price + toc(watch);
                if ~terminal, plan{steps} = tags; end %#ok<AGROW>
            end
            if ~track && boundDB <= o.toleranceDB, break; end
        end
    end
    [eta,A,C,hot,cold,U,timing] = receiver(x,W,track,timing);
    if ~track && 10*log10(eta/entryScore) < o.toleranceDB && boundDB <= o.toleranceDB
        status = "tolerance reached"; break;
    end
end
if partE, timing.firstUserEverActive = nnz(everActive); end
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

function [eta,A,C,hot,cold,U,timing] = receiver(x,W,track,timing)
Q = pagemtimes(W,'none',W,'ctranspose');
U = zeros(x.nr,x.N); A = zeros(x.nt,x.nt,x.N); C = A;
hot = zeros(1,x.N); cold = hot;
for n = 1:x.N
    if track, watch = tic; end
    clutter = kernel_apply(x.V{n},Q(:,:,n),x.nr);
    if track, timing.kernel = timing.kernel + toc(watch); watch = tic; end
    u = mvdr_beamformer(clutter+x.R(:,:,n),x.b{n},0);
    if track, timing.mvdr = timing.mvdr + toc(watch); end
    U(:,n) = u;
    A(:,:,n) = x.sig(n)*abs(u'*x.b{n})^2*(x.a{n}*x.a{n}');
    if track, watch = tic; end
    C(:,:,n) = build_Ac(x.V{n},u,x.nt);
    if track, timing.kernel = timing.kernel + toc(watch); end
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

function [power,dual,v,timing] = allocate_power(x,bank,G,noiseRatio,A,C,hot,backend,track,timing)
% For each candidate (unit) direction, power contributes linearly to
% target return, clutter, and user signals. Change variables z=t*p/Psafe
% and normalize t+clutter'*z=1 to solve the fractional objective as one LP.
if track, watch = tic; end
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
if track, timing.build = timing.build + toc(watch); watch = tic; end
[solution,cost,mult] = solve_lp(backend,objective,ineq,eq);
if track, timing.solve = timing.solve + toc(watch); end
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

function [bank,boundDB,added] = enrich_directions(x,bank,G,dual,v,append,forceAll)
% Leading eigenvector maximizes reduced cost over all unit directions
% for one tone and stream owner. Add it when the current LP can improve.
if nargin < 7, forceAll = false; end
excess = 0;
added = cell(x.N,1);
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
        if append && (forceAll || improvement>1e-7)
            bank{n}.V(:,end+1) = modes(:,idx);
            bank{n}.owner(end+1) = owner;
            added{n}(end+1) = owner;
        end
    end
end
% Reduced-cost bound for the tightened, fixed-receiver allocation problem.
boundDB = 10*log10(1+excess/max(v.ratio,eps));
end

function [bank,inference,pricing] = append_l2o_directions(x,bank,G,noiseRatio,dual,v,model)
inference = 0;
pricing = 0;
for n = 1:x.N
    watch = tic;
    H = G{n};
    mu = dual.user(:,n);
    base = v.matrix(:,:,n) - (H.*mu.')*H';
    [modes,spectrum] = eig((base+base')/2,'vector');
    [~,idx] = max(real(spectrum));
    pricing = pricing + toc(watch);

    watch = tic;
    B = zeros(x.nt,4,x.K,'like',H);
    M = zeros(x.nt,x.nt,x.K,'like',H);
    for k = 1:x.K
        h = H(:,k);
        B(:,:,k) = [bank{n}.V(:,k), h, x.a{n}, bank{n}.V(:,x.K+1)];
        M(:,:,k) = base + (1+1/v.gamma)*mu(k)*(h*h');
    end
    users = parte_l2o_proposals(B,M,H,x.gamma,noiseRatio(:,n),model);
    inference = inference + toc(watch);
    bank{n}.V = [bank{n}.V, users, modes(:,idx)];
    bank{n}.owner = [bank{n}.owner, 1:x.K, 0];
end
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
