function summary = run_l2o_matlab_sweep(tutorial,destination,opts)
%RUN_L2O_MATLAB_SWEEP Time and score the frozen reduced-cost and envelope policies.
% Defaults use the archived sweep grid with the estimated design kernel.
% Optionally add analytic pricing or run both historical design kernels.
arguments
    tutorial (1,1) string
    destination (1,1) string
    opts.Users (1,:) double {mustBeInteger,mustBePositive,mustBeNonempty} = [4 8 12]
    opts.SINRdB (1,:) double {mustBeFinite,mustBeReal,mustBeNonempty} = 0:4:20
    opts.Seeds (1,:) double {mustBeInteger,mustBeNonnegative,mustBeNonempty} = 11001:11008
    opts.DesignKernels (1,:) string {mustBeNonempty,mustBeMember(opts.DesignKernels,["estimated","perfect"])} = "estimated"
    opts.Budget (1,2) double {mustBeInteger,mustBePositive} = [3 2]
    opts.Repeats (1,1) double {mustBeInteger,mustBePositive} = 3
    opts.Threads (1,1) double {mustBeInteger,mustBePositive} = 1
    opts.IncludeAnalytic (1,1) logical = false
end
assert(~isfolder(destination) && ~isfile(destination),'Use a fresh destination.');
assert(numel(unique(opts.Users)) == numel(opts.Users),'User counts must be unique.');
assert(numel(unique(opts.DesignKernels)) == numel(opts.DesignKernels),'Design kernels must be unique.');
assert(prod(opts.Budget) >= 2,'Use at least two LP solves so the learned directions can enter a master.');

runs = cell(numel(opts.Users)*numel(opts.DesignKernels),1);
idx = 0;
for kernel = opts.DesignKernels
    for users = opts.Users
        folder = fullfile(destination,"K"+users,kernel);
        part = run_matlab_sinr_sweep(tutorial,folder,Users=users,SINRdB=opts.SINRdB, ...
            Seeds=opts.Seeds,Budget=opts.Budget,Repeats=opts.Repeats,Threads=opts.Threads, ...
            IncludeAO=false,IncludeAnalytic=opts.IncludeAnalytic,DesignKernel=kernel);
        part.users = repmat(users,height(part),1);
        part.designKernel = repmat(kernel,height(part),1);
        idx = idx+1;
        runs{idx} = movevars(part,{'users','designKernel'},'Before','gammaDB');
    end
end
summary = vertcat(runs{:});
writetable(summary,fullfile(destination,'summary.csv'));
end
