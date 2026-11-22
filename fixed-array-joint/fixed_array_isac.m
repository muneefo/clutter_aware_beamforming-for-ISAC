function out = fixed_array_isac(mode, seed, trials)
% FIXED_ARRAY_ISAC Fixed-array, radiative near-field, monostatic ISAC model.
% out = fixed_array_isac()                 % simulate 128 new MATLAB scenes
% out = fixed_array_isac('simulate',42,5)  % reproducible native simulation
% out = fixed_array_isac('verify')         % compare four shared Python fixtures
% Power, SINR and SCNR values returned by this function are linear.
if nargin < 1, mode = 'simulate'; end
if nargin < 2, seed = 42; end
if nargin < 3, trials = 128; end

% Joint-polytope modes use the separate trained coefficients in joint_model.json.
if strcmpi(mode,'joint')
    out=fixed_array_joint('simulate',seed,trials); return
elseif strcmpi(mode,'verify_joint')
    out=fixed_array_joint('verify'); return
end

p.Nt=24; p.Nr=16; p.K=2; p.L=2; p.B=6; p.S=96;
p.lambda=0.01; p.P=1; p.gamma=2; p.userNoise=6e-7;
p.rxNoise=2e-11; p.beta=1; p.delta=0.05;
% Frozen coefficients trained on a separate Python training split.
theta=[0.48585426240899854; -0.0008094944171145982; 0.0010368262449414907; ...
       0.013803566232063545; 0.1605552128619474; 0.021953524359346943];
schedule=[0.12, 0.12, 0.015];
if strcmpi(mode,'verify')
    file=fullfile(fileparts(mfilename('fullpath')),'reference_fixture.mat');
    f=load(file);
    check_close('initializer',theta,f.theta);
    check_close('schedule',schedule(:),f.schedule(:));
    for i=1:size(f.pos,3)
        data=run_scene(f.pos(:,:,i),f.truth(:,i),f.alpha(:,:,:,i),f.noise(:,:,i),p,theta,schedule);
        assert(~data.empty,'Shared fixture unexpectedly has an empty confidence set.');
        check_close('probe responses H',data.H,f.H(:,:,i));
        check_close('observed energies',data.m,f.m(:,i));
        check_close('estimated clutter powers',data.hat,f.hat(:,i));
        check_close('box maxima',data.upper,f.upper(:,i));
        check_close('all seven designs',data.metrics,f.metrics(:,:,i));
        check_close('complex operating beams',data.W,f.W(:,:,i));
        check_close('bound, true SCNR, user SINRs, power',data.metrics(end,:).',f.values(:,i));
    end
    % Check boundary solutions and explicitly flag an inconsistent calibration.
    H=[1 0;0 1;1 1;2 1;1 2;2 2];
    q=[0;1.5];
    [hat,upper]=infer_powers(H,1+H*q,p);
    check_close('noiseless boundary power recovery',hat,q);
    assert(all(upper>=q-1e-9),'Expected the true powers inside the box.');
    [~,upper]=infer_powers(H,zeros(6,1),p);
    assert(isempty(upper),'Unusable calibration should request recalibration.');
    out=struct('passed',true,'fixtures',size(f.pos,3));
    fprintf('Passed %d shared Python reference fixtures.\n',out.fixtures);
    return
end
if ~strcmpi(mode,'simulate'), error('Use joint, verify_joint, simulate or verify.'); end
validateattributes(trials,{'numeric'},{'scalar','integer','positive'});
rng(seed,'twister');
base=[0.035 0.73; 0.085 0.70; -0.12 0.82; -0.19 0.66; 0.20 0.84];
names={'known_strengths','plug_in','box_classical','box_geometry_classical', ...
       'box_init_only','box_steps_only','box_combined'};
metrics=nan(trials,7,5); covered=false(trials,1); valid=false(trials,1);
heldoutError=nan(trials,1);
for i=1:trials
    pos=base+0.022*(2*rand(size(base))-1);
    truth=exp(-0.8+1.8*rand(p.L,1));
    alpha=complex_normal([p.S,p.B,p.L]); noise=complex_normal([p.S,p.B]);
    data=run_scene(pos,truth,alpha,noise,p,theta,schedule);
    if data.empty, continue; end
    valid(i)=true; covered(i)=data.covered; heldoutError(i)=data.covError;
    metrics(i,:,:)=reshape(data.metrics,[1,7,5]);
end
out=struct(); out.methods=names;
out.columns={'design_scnr','true_scnr','user1_sinr','user2_sinr','power_W'};
out.metrics=metrics; out.valid=valid; out.setCoversTruth=covered;
out.heldoutCovarianceRelativeError=heldoutError;
out.config=p; out.seed=seed;
fprintf('%d scenes, %d feasible returned combined designs, %d empty sets.\n',trials,nnz(valid),nnz(~valid));
if any(valid)
    fprintf('Confidence set covered truth in %d scenes. SCNR bound held in %d returned designs.\n', ...
        sum(covered),sum(metrics(valid,end,1)<=metrics(valid,end,2)+1e-9));
    for i=1:numel(names)
        fprintf('%25s   median true SCNR %7.3f dB\n',names{i},10*log10(median(metrics(valid,i,2))));
    end
end
end

function X=complex_normal(dims)
X=(randn(dims)+1i*randn(dims))/sqrt(2);
end

function a=steering(pos,n,offset,p)
x=((0:n-1).'-(n-1)/2)*p.lambda/2+offset;
distance=hypot(x-pos(:,1).',pos(:,2).');
a=p.lambda./(4*pi*distance).*exp((-2i*pi/p.lambda)*distance);
end

function out=run_scene(pos,truth,alpha,noise,p,theta,schedule)
ta=steering(pos,p.Nt,-0.008,p); ra=steering(pos,p.Nr,0.008,p);
t=ta(:,1:3); r=ra(:,1:3); h=ta(:,4:5).';
G=zeros(p.B,p.L);
for b=1:p.B
    tx=mod(b-1,3)+1; rx=mod(b-1+floor((b-1)/3),3)+1;
    f=conj(t(:,tx))/norm(t(:,tx)); v=r(:,rx)/norm(r(:,rx));
    G(b,:)=(v'*r(:,2:3)).*(t(:,2:3).'*f).';
end
% Clutter-only calibration. The following operation uses the hidden truth
% solely to generate observations. The estimated and robust solvers see H,m.
echo=sqrt(p.rxNoise)*noise;
for j=1:p.L
    echo=echo+alpha(:,:,j).*(sqrt(truth(j))*G(:,j).');
end
H=abs(G).^2/p.rxNoise;
m=mean(abs(echo).^2,1).'/p.rxNoise;
assert(rank(H)==p.L,'Unidentifiable mapped powers: redesign the probe beams.');
[hat,upper,C,d]=infer_powers(H,m,p);
out=struct('empty',isempty(upper),'H',H,'m',m,'hat',hat,'upper',upper);
if out.empty, return; end
out.covered=all(C*truth<=d+1e-9);
zero=zeros(size(theta)); geometryOnly=zero; geometryOnly(1)=1;
qList={truth,hat,upper,upper,upper,upper,upper};
initList={theta,theta,zero,geometryOnly,theta,zero,theta};
stepList={schedule,schedule,0.12*ones(size(schedule)),0.12*ones(size(schedule)),[],schedule,schedule};
metrics=zeros(numel(qList),5);
for j=1:numel(qList)
    W=design_beams(t,r,h,qList{j},initList{j},stepList{j},p);
    [bound,u]=receiver(t,r,W,qList{j},p);
    gain=sum(abs(t.'*W).^2,2).*abs(u'*r).'.^2/p.rxNoise;
    actual=p.beta*gain(1)/(1+truth.'*gain(2:end));
    metrics(j,:)=[bound,actual,sinrs(h,W,p).',norm(W,'fro')^2];
    assert(is_feasible(h,W,p),'Returned precoder violates original constraints.');
end
out.W=W; out.metrics=metrics;
illum=sum(abs(t(:,2:3).'*W).^2,2);
Qclutter=(r(:,2:3).*(illum.*truth).')*r(:,2:3)';
Qestimated=(r(:,2:3).*(illum.*hat).')*r(:,2:3)';
out.covError=norm(Qestimated-Qclutter,'fro')/norm(Qclutter,'fro');
assert(~out.covered || metrics(end,1)<=metrics(end,2)+1e-9,'Bound failed on coverage event.');
end

function [hat,upper,C,d]=infer_powers(H,m,p)
% Exact two-variable NNLS: unconstrained minimizer, the two rays and origin.
y=m-1; candidates=zeros(2,3);
for j=1:2
    candidates(j,j+1)=max(0,(H(:,j)'*y)/(H(:,j)'*H(:,j)));
end
s=H\y;
if all(s>=0), candidates(:,end+1)=s; end
[~,i]=min(sum((H*candidates-y).^2,1)); hat=candidates(:,i);
% If z is circular Gaussian, sum |z|^2 / E|z|^2 is Gamma(S,1).
a=p.delta/(2*p.B);
lo=p.S*m/gammaincinv(1-a,p.S); hi=p.S*m/gammaincinv(a,p.S);
C=[H;-H;-eye(2)]; d=[hi-1;1-lo;zeros(2,1)];
upper=[];
for i=1:size(C,1)-1
    for j=i+1:size(C,1)
        E=C([i,j],:);
        if cond(E)>1e10, continue; end
        v=E\d([i,j]);
        if all(C*v<=d+1e-9)
            if isempty(upper), upper=v; else, upper=max(upper,v); end
        end
    end
end
if ~isempty(upper), upper=max(upper,0); end
end

function [val,u,weights,den]=receiver(t,r,W,q,p)
illum=sum(abs(t.'*W).^2,2);
R=eye(p.Nr)+(r(:,2:3).*(q.*illum(2:3)/p.rxNoise).')*r(:,2:3)';
u=R\r(:,1); u=u/norm(u);
weights=abs(u'*r).'.^2/p.rxNoise;
den=1+sum(q.*weights(2:3).*illum(2:3));
val=p.beta*weights(1)*illum(1)/den;
end

function v=sinrs(h,W,p)
power=abs(h*W).^2; desired=diag(power);
v=desired./(p.userNoise+sum(power,2)-desired);
end

function ok=is_feasible(h,W,p)
ok=norm(W,'fro')^2<=p.P+1e-9 && all(sinrs(h,W,p)>=p.gamma-1e-9);
end

function W=design_beams(t,r,h,q,theta,steps,p)
zf=h'/(h*h');
W0=(zf./sqrt(sum(abs(zf).^2,1)))*sqrt(p.P/p.K);
v=conj(t(:,1))-zf*(h*conj(t(:,1)));
V=(v/norm(v))*exp(1i*angle(t(:,1).'*W0))*sqrt(p.P/p.K);
gamma0=sinrs(h,W0,p);
feat=[1;log1p(q);log(gamma0/p.gamma);norm(t(:,1))*100];
f=min(max(feat.'*theta,0),0.94);
f=min(f,max(0,0.995*(1-p.gamma/min(gamma0))));
W=sqrt(1-f)*W0+sqrt(f)*V;
if ~is_feasible(h,W,p), W=W0; end
assert(is_feasible(h,W,p),'Zero-forcing initializer infeasible on this geometry.');
for eta=steps(:).'
    [val,~,weights,den]=receiver(t,r,W,q,p);
    coeff=[p.beta*weights(1);-val*q.*weights(2:3)]/den;
    grad=(conj(t).*coeff.')*(t.'*W);
    if norm(grad,'fro')<1e-15, continue; end
    direction=grad/norm(grad,'fro');
    for j=0:8
        raw=W+eta*2^(-j)*direction;
        candidate=sqrt(p.P)*raw/norm(raw,'fro');
        if is_feasible(h,candidate,p) && receiver(t,r,candidate,q,p)>=val-1e-12
            W=candidate; break
        end
    end
end
end

function check_close(label,x,y)
assert(isequal(size(x),size(y)),'Shape mismatch for %s.',label);
errorMax=max(abs(x(:)-y(:)));
tolerance=2e-8*max(1,max(abs(y(:))));
assert(isfinite(errorMax) && errorMax<=tolerance, ...
    '%s: max abs difference %.3g exceeds %.3g.',label,errorMax,tolerance);
end
