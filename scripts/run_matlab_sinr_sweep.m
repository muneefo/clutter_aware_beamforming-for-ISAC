function summary = run_matlab_sinr_sweep(tutorial,destination,opts)
%RUN_MATLAB_SINR_SWEEP Paired MATLAB design timing and SCNR threshold sweep.
% Use a fresh MATLAB process and output folder. Disable both optional
% references for the two frozen L2O policies alone.
arguments
    tutorial (1,1) string
    destination (1,1) string
    opts.SINRdB (1,:) double {mustBeFinite,mustBeReal,mustBeNonempty} = 4:2:14
    opts.Seeds (1,:) double {mustBeInteger,mustBeNonnegative,mustBeNonempty} = 16001:16032
    opts.Users (1,1) double {mustBeInteger,mustBePositive} = 4
    opts.Budget (1,2) double {mustBeInteger,mustBePositive} = [3 2]
    opts.Repeats (1,1) double {mustBeInteger,mustBePositive} = 3
    opts.Threads (1,1) double {mustBeInteger,mustBePositive} = 1
    opts.IncludeAO (1,1) logical = true
    opts.IncludeAnalytic (1,1) logical = true
    opts.DesignKernel (1,1) string {mustBeMember(opts.DesignKernel,["estimated","perfect"])} = "estimated"
end
assert(~isfolder(destination) && ~isfile(destination),'Use a fresh destination.');
assert(numel(unique(opts.SINRdB)) == numel(opts.SINRdB),'Thresholds must be unique.');
assert(numel(unique(opts.Seeds)) == numel(opts.Seeds) && all(opts.Seeds < 2^32), ...
    'Seeds must be distinct and fit in 32 bits.');
assert(exist('linprog','file') ~= 0,'Optimization Toolbox linprog is required.');
if opts.IncludeAO
    assert(exist('cvx_begin','file') ~= 0 && exist('mosekopt','file') ~= 0, ...
        'AO-SDP requires configured CVX and MOSEK. For LPs only, set IncludeAO=false.');
end
oldPath = path;
pathCleanup = onCleanup(@() path(oldPath)); %#ok<NASGU>
oldRng = rng;
rngCleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
oldThreads = maxNumCompThreads(opts.Threads);
threadCleanup = onCleanup(@() maxNumCompThreads(oldThreads)); %#ok<NASGU>
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'first-isac-paper','experiments'));
addpath(fullfile(tutorial,'function'));

[covPath,revision,covHash] = pinned_inputs(tutorial);
data = load(covPath,'para','Vcc_hat','Vcc','R_eta');
if opts.DesignKernel == "perfect", data.Vcc_hat = data.Vcc; end
assert(opts.Users < min(data.para.Nt,data.para.Ns),'The ZF start needs a spare dimension.');
checkpoint = fullfile(root,'results','matlab-sweep-2026-11-23','l2o-weights.mat');
weights = load(checkpoint);
check_forward_pass(weights);
methods = ["reducedCost","envelope"];
if opts.IncludeAnalytic, methods = ["analytic",methods]; end
if opts.IncludeAO
    cvx_solver mosek
    cvx_solver_settings('MSK_IPAR_NUM_THREADS',opts.Threads);
    methods = ["ao",methods];
end

mkdir(destination);
manifest = struct('options',opts,'methods',methods,'matlab',version,'platform',computer, ...
    'blas',version('-blas'),'toolboxes',ver,'matlabThreads',maxNumCompThreads, ...
    'linprog','explicit primal/dual tolerances 1e-9', ...
    'harnessSHA256',sha256(which('run_matlab_sinr_sweep')), ...
    'solverSHA256',sha256(which('fast_tutorial_beams')), ...
    'forwardSHA256',sha256(which('parte_l2o_proposals')), ...
    'tutorialRevision',revision,'covarianceSHA256',covHash, ...
    'checkpointSHA256',sha256(checkpoint),'reducedCostSHA256',weights.reducedCost.sourceSHA256, ...
    'envelopeSHA256',weights.envelope.sourceSHA256,'warmupSeed',15999, ...
    'orderSeed',606231,'bootstrapSeed',606232,'bootstrapDraws',2000);
while ismember(manifest.warmupSeed,opts.Seeds)
    manifest.warmupSeed = manifest.warmupSeed - 1;
end
if isunix
    [~,manifest.os] = system('uname -a');
    [~,manifest.cpu] = system('lscpu');
elseif ispc
    manifest.cpu = getenv('PROCESSOR_IDENTIFIER');
end
if opts.IncludeAO
    manifest.cvx = evalc('cvx_version');
    manifest.mosek = evalc('mosekdiag');
    manifest.mosekThreads = opts.Threads;
end
write_json(fullfile(destination,'environment.json'),manifest);

orderStream = RandStream('mt19937ar','Seed',manifest.orderSeed);
rawPath = fullfile(destination,'raw.csv');
warmup = repmat(empty_row(),numel(methods),1);
for i = 1:numel(methods)
    warmup(i) = attempt(tutorial,data,weights,opts,methods(i),manifest.warmupSeed,opts.SINRdB(1),0);
end
writetable(struct2table(warmup),fullfile(destination,'warmup.csv'));
for gammaDB = opts.SINRdB
    for seed = opts.Seeds
        rows = repmat(empty_row(),opts.Repeats*numel(methods),1);
        ix = 0;
        for rep = 1:opts.Repeats
            order = randperm(orderStream,numel(methods));
            for j = order
                ix = ix+1;
                rows(ix) = attempt(tutorial,data,weights,opts,methods(j),seed,gammaDB,rep);
            end
        end
        if ~isfile(rawPath)
            writetable(struct2table(rows),rawPath);
        else
            writetable(struct2table(rows),rawPath,'WriteMode','append','WriteVariableNames',false);
        end
        fprintf('SINR %g dB, seed %d: %d/%d feasible runs\n', ...
            gammaDB,seed,nnz([rows.feasible]),numel(rows));
    end
end
summary = plot_matlab_sinr_sweep(destination);
end

function row = attempt(tutorial,data,weights,opts,method,seed,gammaDB,rep)
row = empty_row();
row.method = method;
row.seed = seed;
row.gammaDB = gammaDB;
row.rep = rep;
watch = tic;
try
    if method == "ao"
        row = run_ao(row,data,opts.Users);
    else
        trial = table("AO_SDP",seed,NaN,NaN,false,'VariableNames', ...
            {'method','scene','seconds','trueSCNRdB','feasible'});
        request = struct('options',struct('K',opts.Users,'gammaDB',gammaDB,'qualityGapDB',NaN), ...
            'trials',trial);
        if method ~= "analytic", request.model = weights.(char(method)); end
        out = fast_tutorial_beams(tutorial,request,struct('outer',opts.Budget(1), ...
            'inner',opts.Budget(2),'matchedPolicy',method,'quiet',true),data);
        r = out.trials;
        t = out.timing{1};
        row.designSeconds = r.seconds;
        row.designSCNRdB = r.designSCNRdB;
        row.trueSCNRdB = r.trueSCNRdB;
        row.minMarginDB = r.minMarginDB;
        row.powerW = r.powerW;
        row.feasible = r.feasible;
        row.lpSolves = r.lpSolves;
        row.appendedAtoms = t.appendedAtoms;
        row.firstUserProposals = t.firstUserProposals;
        row.firstUserEverActive = t.firstUserEverActive;
        row.terminalBoundDB = t.terminalBoundDB;
        for key = ["build","solve","price","mvdr","kernel","recovery","inference"]
            row.(char(key+"Seconds")) = t.(char(key));
        end
        assert(row.lpSolves == prod(opts.Budget) && ...
            row.appendedAtoms == 32*(opts.Users+1)*(prod(opts.Budget)-1),'Mismatched Part E schedule.');
    end
    row.feasible = row.feasible && all(isfinite([row.designSCNRdB,row.trueSCNRdB,row.designSeconds])) ...
        && row.designSeconds > 0;
    row.status = "returned";
catch err
    row.feasible = false;
    row.status = "error";
    row.message = string(regexprep(err.message,'[\r\n]+',' '));
end
row.attemptSeconds = toc(watch);
end

function row = run_ao(row,data,K)
para = data.para;
para.N = 32;
para.L = 64;
para.Pt = 100;
para.Ku = K;
N = para.N;
Vhat = data.Vcc_hat(1:N);
V = data.Vcc(1:N);
R = data.R_eta(:,:,1:N);
gamma = 10^(row.gammaDB/10);
noise = 1e-19*para.deltaf;
f = para.fc + (0:N-1)*para.deltaf;
steer = @(n) exp(-1i*pi*(0:n-1)'*sind(-10).*(f/para.fc))/sqrt(n);
A = num2cell(steer(para.Nt),1);
B = num2cell(steer(para.Nr),1);
target = para.Pt*(para.c0./f).^2*1e4*10^.5/((4*pi)^3*41.8^4);
rng(row.seed,'twister');
H = (randn(para.Nt,N,K) + 1i*randn(para.Nt,N,K))/sqrt(2);
dist = randi([50,400],1,1,K);
H = H .* 10.^((-30-27*log10(dist))/20);
rng(mod(17389+1664525*row.seed,2^32),'twister');
settings = struct('maxAO',30,'maxDinkel',30,'tolAO_dB',0.01, ...
    'tolDinkel_eta',0.001,'lambdaRidge',0,'verbose',false);
watch = tic;
sol = solve_BLP_AO(para,target,Vhat,R,A,B,H,gamma*ones(N,K),noise,settings);
row.designSeconds = toc(watch);
W = cat(2,sol.w_comm,sol.Ws);
Q = pagemtimes(W,'none',W,'ctranspose');
row.designSCNRdB = 10*log10(compute_SCNR(Q,sol.u,Vhat,R,target,A,B));
row.trueSCNRdB = 10*log10(compute_SCNR(Q,sol.u,V,R,target,A,B));
row.minMarginDB = inf;
for n = 1:N
    y = abs(reshape(H(:,n,:),para.Nt,K)'*W(:,:,n)).^2;
    desired = diag(y(:,1:K));
    y(1:K+1:K*K) = 0;
    row.minMarginDB = min(row.minMarginDB,min(10*log10(desired./((sum(y,2)+noise)*gamma))));
end
row.powerW = sum(abs(W).^2,'all');
row.feasible = row.minMarginDB >= -1e-4 && row.powerW <= para.Pt*(1+1e-7);
row.aoUpdates = numel(sol.eta_hist)-1;
end

function row = empty_row()
row = struct('method',"",'seed',0,'gammaDB',0,'rep',0,'status',"",'message',"", ...
    'designSeconds',NaN,'attemptSeconds',NaN,'designSCNRdB',NaN,'trueSCNRdB',NaN, ...
    'minMarginDB',NaN,'powerW',NaN,'feasible',false,'aoUpdates',NaN,'lpSolves',NaN, ...
    'appendedAtoms',NaN,'firstUserProposals',NaN,'firstUserEverActive',NaN, ...
    'terminalBoundDB',NaN,'buildSeconds',NaN,'solveSeconds',NaN,'priceSeconds',NaN, ...
    'mvdrSeconds',NaN,'kernelSeconds',NaN,'recoverySeconds',NaN,'inferenceSeconds',NaN);
end

function check_forward_pass(weights)
f = weights.fixture;
for name = ["reducedCost","envelope"]
    [v,features] = parte_l2o_proposals(f.basis,f.matrices,f.channels,f.gamma,f.noiseRatio,weights.(char(name)));
    assert(max(abs(v-f.(char(name))),[],'all') < 1e-10 && ...
        max(abs(features-f.features),[],'all') < 1e-10,'L2O parity fixture failed.');
end
end

function [source,revision,digest] = pinned_inputs(tutorial)
if ispc
    assert(isempty(regexp(tutorial,'["%!?&|^<>\r\n]','once')),'Unsupported tutorial path.');
    quoted = """" + tutorial + """";
else
    q = string(char(39));
    assert(~any(contains(tutorial,[q,string(char(10)),string(char(13))])),'Unsupported tutorial path.');
    quoted = q + tutorial + q;
end
[code,revision] = system("git -C "+quoted+" rev-parse HEAD");
revision = strtrim(revision);
assert(code == 0 && strcmp(revision,'0f864c45b4775fbe01789b313d81f22bebf65177'), ...
    'Check out the pinned tutorial revision.');
source = fullfile(tutorial,'data','covMat_BLP.mat');
digest = sha256(source);
assert(strcmp(digest,'99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d'), ...
    'Unexpected tutorial covariance file.');
end

function value = sha256(filename)
fid = fopen(filename,'rb');
assert(fid >= 0,'Could not read %s.',filename);
closeFile = onCleanup(@() fclose(fid)); %#ok<NASGU>
digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(typecast(fread(fid,inf,'*uint8'),'int8'));
value = lower(reshape(dec2hex(typecast(digest.digest(),'uint8')).',1,[]));
end

function write_json(filename,data)
fid = fopen(filename,'w');
assert(fid >= 0,'Could not write %s.',filename);
closeFile = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));
end
