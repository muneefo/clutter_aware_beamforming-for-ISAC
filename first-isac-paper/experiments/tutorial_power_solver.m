function solver = tutorial_power_solver(backends,threads)
%TUTORIAL_POWER_SOLVER Select a power LP backend with a checked inequality price.
% The two-variable LP has z = [0.5; 0.5] and inequality multiplier 0.5.
if nargin < 2, threads = 1; end
failures = strings(0,1);
for backend = backends
    solve = @(f,ineq,eq) solve_lp(backend,f,ineq,eq,threads);
    try
        [z,cost,price,algorithm] = solve([-1;0],sparse([1,-1]),sparse([1,1]));
        assert(norm(z-[0.5;0.5],inf)<1e-5 && abs(cost+0.5)<1e-5 ...
            && isscalar(price) && abs(price-0.5)<1e-5, ...
            'Unexpected primal solution or inequality multiplier.');
        solver = struct('solve',solve,'backend',backend,'algorithm',algorithm, ...
            'tolerances',"solver defaults",'startupFailures',failures);
        if backend == "linprog" || backend == "mosekopt"
            solver.tolerances = "explicit primal/dual tolerances 1e-9";
        end
        return;
    catch err
        failures(end+1,1) = backend + ": " + string(err.message); %#ok<AGROW>
    end
end
error('ISAC:PowerLPStartup','No LP backend passed the startup test.\n%s',join(failures,newline));
end

function [z,cost,mult,algorithm] = solve_lp(backend,f,ineq,eq,threads)
rhs = zeros(size(ineq,1),1);
if backend == "mosekopt"
    % Upper-bounded inequalities followed by one equality. suc holds the
    % nonnegative upper-bound prices used to form the L2O pricing matrices.
    prob.c = f(:);
    prob.a = sparse([ineq;eq]);
    prob.blc = [-inf(size(rhs));1];
    prob.buc = [rhs;1];
    prob.blx = zeros(numel(f),1);
    prob.bux = inf(numel(f),1);
    param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_DUAL_SIMPLEX';
    param.MSK_IPAR_NUM_THREADS = threads;
    param.MSK_DPAR_BASIS_TOL_X = 1e-9;
    param.MSK_DPAR_BASIS_TOL_S = 1e-9;
    [rcode,res] = mosekopt('minimize echo(0)',prob,param);
    if rcode ~= 0
        error('ISAC:MosekLP','mosekopt returned %d (%s).',rcode,res.rcodestr);
    end
    bas = res.sol.bas;
    assert(strcmp(bas.solsta,'OPTIMAL'),'MOSEK basic solution status: %s',bas.solsta);
    z = bas.xx(:);
    cost = prob.c.'*z;
    mult = bas.suc(1:numel(rhs));
    algorithm = "MOSEK dual simplex";
elseif backend ~= "CVX"
    if backend == "linprog"
        opts = optimoptions('linprog','Display','none','ConstraintTolerance',1e-9, ...
            'OptimalityTolerance',1e-9);
        [z,cost,flag,output,lambda] = linprog(f,ineq,rhs,eq,1,zeros(size(f)),[],opts);
    else
        [~,z,cost,flag,output,lambda] = evalc('linprog(f,ineq,rhs,eq,1,zeros(size(f)),[])');
    end
    assert(flag>0,'Power LP failed: %s',output.message);
    mult = lambda.ineqlin;
    if nargout > 3, algorithm = string(output.algorithm); end
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
    cost = cvx_optval; mult = price; algorithm = "CVX";
end
end
