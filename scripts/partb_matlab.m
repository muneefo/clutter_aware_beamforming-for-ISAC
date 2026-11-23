function partb_matlab(tutorial, destination, threads)
%PARTB_MATLAB Benchmark original CVX B1 and native MATLAB LP B6, B7, B8.
% Use a fresh MATLAB process and a fresh results directory for each thread cap.
arguments
    tutorial (1,1) string
    destination (1,1) string
    threads (1,1) double {mustBeMember(threads,[1,2])}
end
assert(~isfolder(destination) && ~isfile(destination), 'Use a fresh destination.');
assert(exist('cvx_begin','file') ~= 0 && exist('mosekopt','file') ~= 0, ...
    'This harness requires configured MATLAB, CVX and MOSEK.');
assert(exist('linprog','file') ~= 0, 'Optimization Toolbox linprog is required.');
old_path = path;
path_cleanup = onCleanup(@() path(old_path)); %#ok<NASGU>
old_threads = maxNumCompThreads(threads);
thread_cleanup = onCleanup(@() maxNumCompThreads(old_threads)); %#ok<NASGU>
cvx_solver mosek
cvx_solver_settings('MSK_IPAR_NUM_THREADS',threads);

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'first-isac-paper','experiments'));
addpath(fullfile(tutorial,'function'));
[git_status, commit] = system("git -C " + quoted(tutorial) + " rev-parse HEAD");
assert(git_status == 0 && strcmp(strtrim(commit),'0f864c45b4775fbe01789b313d81f22bebf65177'), ...
    'Check out the pinned tutorial revision.');
source = fullfile(tutorial,'data','covMat_BLP.mat');
digest = checksum(source);
assert(strcmp(digest,'99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d'), ...
    'Unexpected covariance file.');
mkdir(destination);
[~, system_info] = system('uname -a');
[~, cpu_info] = system('lscpu');
manifest = struct('matlab',version,'blas',version('-blas'),'toolboxes',ver,'cvx',evalc('cvx_version'), ...
    'solver','MOSEK','mosek',evalc('mosekdiag'),'linprog','Optimization Toolbox linprog', ...
    'matlab_thread_cap',maxNumCompThreads,'mosek_thread_cap',threads, ...
    'os',system_info,'cpu',cpu_info,'tutorial_revision',strtrim(commit),'covariance_sha256',digest);
fid = fopen(fullfile(destination,'environment.json'),'w');
assert(fid >= 0, 'Could not write manifest.');
fprintf(fid,'%s\n',jsonencode(manifest,'PrettyPrint',true));
fclose(fid);

data = load(source,'para','Vcc_hat','Vcc','R_eta');
para = data.para;
para.N = 32; para.L = 64; para.Pt = 100; K = 4; para.Ku = K;
gamma = 10^0.8 * ones(para.N,K);
noise = 1e-19*para.deltaf;
f = para.fc + (0:para.N-1)*para.deltaf;
steer = @(n) exp(-1i*pi*(0:n-1)'*sind(-10).*(f/para.fc))/sqrt(n);
A = num2cell(steer(para.Nt),1);
B = num2cell(steer(para.Nr),1);
target = para.Pt*(para.c0./f).^2*1e4*10^.5/((4*pi)^3*41.8^4);
opts = struct('maxAO',30,'maxDinkel',30,'tolAO_dB',0.01,'tolDinkel_eta',0.001, ...
    'lambdaRidge',0,'verbose',false);

% Diagnostic profiler run is deliberately outside the measured traversals.
profile clear
profile on -detail builtin
run_reference(15999, para, K, gamma, noise, A, B, target, data, opts);
profile off
prof = profile('info');
functions = struct2table(prof.FunctionTable,'AsArray',true);
functions.SelfTime = functions.TotalTime;
for k = 1:numel(prof.FunctionTable)
    children = prof.FunctionTable(k).Children;
    if ~isempty(children)
        functions.SelfTime(k) = max(0,functions.TotalTime(k) - sum([children.TotalTime]));
    end
end
writetable(functions(:,{'FunctionName','NumCalls','TotalTime','SelfTime'}), ...
    fullfile(destination,'warmup-profiler-inclusive.csv'));

budget = [3 2; 5 3];
arms = ["B1","B6","B7","B8","B6","B7","B8"];
for seed = [15999,16001:16032]
    reference = table(seed,NaN,NaN,false, ...
        'VariableNames',{'scene','seconds','trueSCNRdB','feasible'});
    placeholder = struct('options',struct('K',K,'gammaDB',8,'qualityGapDB',0.561),'trials',reference);
    schedules = cell(1,2);
    prep = zeros(1,2);
    for b = 1:2
        timer = tic;
        try
            prior = fast_tutorial_beams(tutorial,placeholder, ...
                struct('outer',budget(b,1),'inner',budget(b,2),'ablation',"B8"));
            schedules{b} = prior.pricingPlans{1};
        catch err
            fprintf(2,'Schedule preparation failed for seed %d: %s\n',seed,err.message);
        end
        prep(b) = toc(timer);
    end
    order = randperm(RandStream('mt19937ar','Seed',400000+seed),numel(arms));
    rows = repmat(empty_row(),0,1);
    for i = order
        b = 1 + (i > 4);
        try
            if i == 1
                row = run_reference(seed,para,K,gamma,noise,A,B,target,data,opts);
            else
                assert(arms(i) ~= "B7" || ~isempty(schedules{b}), 'B7 requires a matched B8 schedule.');
                setting = struct('outer',budget(b,1),'inner',budget(b,2), ...
                    'ablation',arms(i),'pricingSchedules',{{schedules{b}}});
                out = fast_tutorial_beams(tutorial,placeholder,setting);
                assert(out.backend == "linprog", 'Unexpected LP backend.');
                if arms(i) ~= "B6" && ~isempty(schedules{b})
                    assert(isequal(out.pricingPlans{1},schedules{b}), 'Mismatched atom schedule.');
                end
                W = out.designs{1}.W; U = out.designs{1}.U;
                Q = pagemtimes(W,'none',W,'ctranspose');
                row = empty_row();
                row.arm = arms(i);
                row.outer = budget(b,1); row.inner = budget(b,2);
                row.seconds = out.trials.seconds;
                row.design_db = 10*log10(compute_SCNR(Q,U,data.Vcc_hat,data.R_eta,target,A,B));
                row.true_db = out.trials.trueSCNRdB;
                row.feasible = out.trials.feasible;
                row.status = "returned";
                row.solves = out.trials.lpSolves;
                timing = out.timing{1};
                for field = ["build","solve","price","mvdr","kernel","recovery"]
                    row.(char(field+"_seconds")) = timing.(char(field));
                end
                row.schedule_prep_seconds = prep(b);
            end
        catch err
            row = empty_row();
            row.arm = arms(i);
            if i == 1
                row.outer = 30; row.inner = 30;
            else
                row.outer = budget(b,1); row.inner = budget(b,2);
                row.schedule_prep_seconds = prep(b);
            end
            row.status = "error";
            row.error_message = string(err.message);
        end
        row.seed = seed;
        row.warmup = seed == 15999;
        rows(end+1,1) = row; %#ok<AGROW>
    end
    writetable(struct2table(rows),fullfile(destination,"scene-" + seed + ".csv"));
end
end

function row = run_reference(seed, para, K, gamma, noise, A, B, target, data, opts)
rng(seed,'twister');
H = (randn(para.Nt,para.N,K) + 1i*randn(para.Nt,para.N,K))/sqrt(2);
dist = randi([50,400],1,1,K);
H = H.*10.^((-30-27*log10(dist))/20);
rng(mod(17389+1664525*seed,2^32),'twister');
timer = tic;
sol = solve_BLP_AO(para,target,data.Vcc_hat,data.R_eta,A,B,H,gamma,noise,opts);
seconds = toc(timer);
Q = sol.RX;
W = cat(2,sol.w_comm,sol.Ws);
U = sol.u;
physical = pagemtimes(W,'none',W,'ctranspose');
margin = inf;
for n = 1:para.N
    channel = reshape(H(:,n,:),para.Nt,K);
    y = abs(channel'*W(:,:,n)).^2;
    desired = diag(y(:,1:K));
    y(1:K+1:K*K) = 0;
    margin = min(margin,min(10*log10(desired./((sum(y,2)+noise).*gamma(n,:).'))));
end
row = empty_row();
row.arm = "B1"; row.outer = 30; row.inner = 30; row.status = "returned";
row.seconds = seconds;
row.design_db = 10*log10(compute_SCNR(physical,U,data.Vcc_hat,data.R_eta,target,A,B));
row.true_db = 10*log10(compute_SCNR(physical,U,data.Vcc,data.R_eta,target,A,B));
row.cov_design_db = 10*log10(compute_SCNR(Q,U,data.Vcc_hat,data.R_eta,target,A,B));
row.cov_true_db = 10*log10(compute_SCNR(Q,U,data.Vcc,data.R_eta,target,A,B));
row.feasible = margin >= -1e-4 && sum(abs(W).^2,'all') <= para.Pt*(1+1e-7);
end

function row = empty_row()
row = struct('arm',"",'outer',0,'inner',0,'seed',0,'warmup',false,'seconds',NaN, ...
    'design_db',NaN,'true_db',NaN,'feasible',false,'solves',NaN,'cov_design_db',NaN,'cov_true_db',NaN, ...
    'status',"",'error_message',"",'build_seconds',NaN,'solve_seconds',NaN,'price_seconds',NaN,'mvdr_seconds',NaN, ...
    'kernel_seconds',NaN,'recovery_seconds',NaN,'schedule_prep_seconds',NaN);
end

function value = checksum(filename)
fid = fopen(filename,'rb');
assert(fid >= 0, 'Could not open the benchmark covariance file.');
clean = onCleanup(@() fclose(fid)); %#ok<NASGU>
digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(typecast(fread(fid,inf,'*uint8'),'int8'));
value = lower(reshape(dec2hex(typecast(digest.digest(),'uint8')).',1,[]));
end

function arg = quoted(value)
if ispc
    assert(isempty(regexp(value,'["%!?&|^<>\r\n]','once')), 'Unsupported shell character in repository path.');
    arg = """" + value + """";
else
    quote = string(char(39));
    assert(~any(contains(value,[quote,string(char(10)),string(char(13))])), ...
        'Use a repository path without single quotes or line breaks.');
    arg = quote + value + quote;
end
end
