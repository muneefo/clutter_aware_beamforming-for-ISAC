function T=compare_beam_repairs(repo,outWU,scene)
% Compare three one-step repairs at each state of the recorded anchor run.
% T=compare_beam_repairs(repo,outWU,112)
if nargin<3, scene=112; end
o=outWU.options; net=outWU.model;
saved=outWU.layers(outWU.layers.scene==scene & outWU.layers.method=="learnW_exactU",:);
assert(height(saved)==o.depth,'Select a completed learnW_exactU scene.');
saved=sortrows(saved,'layer');
old=path; restore=onCleanup(@()path(old)); %#ok<NASGU>
oldrng=rng; restoreR=onCleanup(@()rng(oldrng)); %#ok<NASGU>
addpath(fullfile(repo,'function'));
s=load(fullfile(repo,'data','covMat_BLP.mat'),'para','Vcc_hat','R_eta');
p=s.para; p.N=32; p.L=64; p.Pt=100; p.Ku=o.K;
d.p=p; d.V=s.Vcc_hat(1:p.N); d.R=s.R_eta(:,:,1:p.N);
d.noise=1e-19*p.deltaf; d.gamma=10^(o.gammaDB/10)*ones(p.N,p.Ku);
f=p.fc+(0:p.N-1)*p.deltaf;
d.sig=p.Pt*(p.c0./f).^2*1e4*10^.5/((4*pi)^3*41.8^4);
for n=1:p.N
    d.a{n,1}=exp(-1i*pi*(0:p.Nt-1)'*sind(-10)*f(n)/p.fc)/sqrt(p.Nt);
    d.b{n,1}=exp(-1i*pi*(0:p.Nr-1)'*sind(-10)*f(n)/p.fc)/sqrt(p.Nr);
end
x=channels(d,scene); [anchor,W]=start(x,beamseed(scene));
blank=struct('layer',0,'method',"",'rawGainDB',NaN,'gainDB',NaN,'minMarginDB',NaN, ...
    'powerW',NaN,'feasible',false,'wouldAccept',false,'alpha',NaN,'seconds',NaN,'status',"");
rows=repmat(blank,0,1); methods=["anchor","current","nearest"];
for j=1:o.depth
    [~,A,C,noise,cold]=receiverblock(x,W,[],"exact",net);
    eta=beamratio(W,A,C,noise);
    raw=transmitblock(x,W,A,C,eta,noise,cold,j,o.depth,net.W,net.fixed);
    rawgain=10*log10(beamratio(raw,A,C,noise)/eta);
    assert(abs(rawgain-saved.rawGainDB(j))<1e-5,'Replay mismatch; check driver version and inputs.');
    next=W;
    for method=methods
        row=blank; row.layer=j; row.method=method; row.rawGainDB=rawgain;
        timer=tic; Y=alignphases(raw,x);
        try
            if method=="nearest"
                [Y,row.status]=nearest(x,Y);
            else
                Y=Y/max(1,norm(Y(:))/sqrt(p.Pt));
                base=anchor; if method=="current", base=W; end
                [Y,row.alpha]=segment(x,base,Y); row.status="computed";
            end
            row.seconds=toc(timer);
            if all(isfinite(Y(:)))
                [row.feasible,row.minMarginDB,row.powerW]=audit(x,Y);
                v=beamratio(Y,A,C,noise); row.gainDB=10*log10(v/eta);
                row.wouldAccept=row.feasible && v>=eta;
                if method=="anchor"
                    assert(abs(row.gainDB-saved.repairedGainDB(j))<1e-5,'Anchor gain mismatch');
                    if row.wouldAccept, next=Y; end
                end
            end
        catch err
            if method~="nearest", rethrow(err); end
            row.status=string(err.message); row.seconds=toc(timer);
        end
        rows(end+1,1)=row;
    end
    W=next; % Keep all subsequent comparisons on the original trajectory.
end
T=struct2table(rows); disp(T)
end

function [Z,t]=segment(x,base,Y)
if all(sinrs(x,Y)>=x.gamma,'all'), Z=Y; t=1; return; end
lo=0; hi=1;
for it=1:23
    t=(lo+hi)/2; Z=(1-t)*base+t*Y;
    if all(sinrs(x,Z)>=x.gamma,'all'), lo=t; else, hi=t; end
end
t=lo; Z=(1-t)*base+t*Y;
end

function [Z,status]=nearest(x,Y)
% Nearest beam in the phase-fixed convex set. Use unit-noise scaling.
p=x.p; nt=p.Nt; ns=p.Ns; N=p.N;
cvx_begin quiet
    variable Z(nt,ns,N) complex
    minimize(norm(Z(:)-Y(:),2))
    subject to
        norm(Z(:),2)<=sqrt(p.Pt);
        for n=1:N
            for k=1:p.Ku
                h=x.H(:,n,k)/sqrt(x.noise);
                row=h'*Z(:,:,n);
                imag(row(k))==0;
                norm([row([1:k-1,k+1:ns]),1],2)<=real(row(k))/sqrt(x.gamma(n,k));
            end
        end
cvx_end
status=string(cvx_status);
if ~contains(status,"Solved"), Z(:)=NaN; end
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


function Q=covs(W)
Q=pagemtimes(W,'none',W,'ctranspose');
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


function y=blockmlp(w,f,D)
% Shared per-tone 4 -> 6 tanh -> D logits. f has four real entries.
A=reshape(w(1:24),6,4); h=tanh(A*f+w(25:30));
B=reshape(w(31:30+6*D),D,6);
y=B*h+w(31+6*D:end);
end


function [u,At,Ac,noise,cold]=receiverblock(x,W,prev,mode,net)
N=x.p.N; u=zeros(x.p.Nr,N); Q=covs(W);
At=zeros(x.p.Nt,x.p.Nt,N); Ac=At; noise=zeros(1,N); cold=noise;
for n=1:N
    C=kernel_apply(x.V{n},Q(:,:,n),x.p.Nr); R=C+x.R(:,:,n); b=x.b{n};
    if mode=="exact"
        v=mvdr_beamformer(R,b,0);
    else
        old=[]; if ~isempty(prev), old=prev(:,n); end
        weights=net.U;
        if mode=="iter", weights=zeros(size(weights)); end
        v=receive_steps(R,b,old,weights,net.rxSteps);
    end
    u(:,n)=v; noise(n)=real(v'*x.R(:,:,n)*v); cold(n)=real(v'*C*v);
    At(:,:,n)=x.sig(n)*abs(v'*b)^2*(x.a{n}*x.a{n}');
    Ac(:,:,n)=build_Ac(x.V{n},v,x.p.Nt);
end
end


function v=receive_steps(R,b,previous,w,S)
% Minimize v'*R*v on b'*v=1 by projected, preconditioned descent.
b=b(:); bb=real(b'*b);
if isempty(previous), v=b/bb; else, v=previous; end
v=v+b*((1-b'*v)/bb);
diagR=real(diag(R)); scale=mean(diagR);
assert(isfinite(scale)&&scale>0 && all(diagR>0),'Invalid disturbance diagonal.');
R=R/scale; diagR=diagR/scale;
for i=1:S
    y=R*v; r=y-b*((b'*y)/bb);
    if norm(r)<=1e-12*max(norm(y),realmin), break; end
    f=[norm(r)/max(norm(y),realmin);tanh(log(max(diagR)/min(diagR))); ...
       tanh(log(max(norm(v)*sqrt(bb),realmin))); i/S];
    z=blockmlp(w,f,2); mix=(1+tanh(z(1)))/2;
    dir=(1-mix)*r+mix*(r./diagR);
    dir=dir-b*((b'*dir)/bb);
    denom=real(dir'*R*dir);
    if denom<=realmin, break; end
    rate=real(dir'*r)/denom;
    v=v-(1+.8*tanh(z(2)))*rate*dir;
end
v=v+b*((1-b'*v)/bb);
end


function s=beamratio(W,A,C,noise)
num=real(sum(conj(W).*pagemtimes(A,W),'all'));
den=sum(noise)+real(sum(conj(W).*pagemtimes(C,W),'all'));
s=num/den; assert(isfinite(s) && s>0 && den>0,'Invalid fixed-filter SCNR.');
end


function Y=transmitblock(x,W,A,C,eta,noise,cold,j,D,w,baseStep)
N=x.p.N; K=x.p.Ku; Y=W; ratios=sinrs(x,W)./x.gamma;
for n=1:N
    Wn=W(:,:,n); H=reshape(x.H(:,n,:),x.p.Nt,K);
    % Real part of the complex inner product with grad gives the local
    % change in log SINR (up to a common factor of two).
    qos=zeros(size(Wn)); hW=H'*Wn;
    for k=1:K
        row=hW(k,:); d=row(k); row(k)=0;
        grad=-H(:,k)*row/(sum(abs(row).^2)+x.noise);
        grad(:,k)=H(:,k)*d/max(abs(d)^2,realmin);
        qos=qos+grad/max(norm(grad,'fro'),realmin)/max(1,ratios(n,k))^2;
    end
    f=[cold(n)/max(cold(n)+noise(n),realmin); ...
       tanh(log(max(min(ratios(n,:)),realmin))); ...
       tanh(log(max(norm(Wn,'fro')^2*N/x.p.Pt,realmin)));j/D];
    z=blockmlp(w,f,4);
    target=A(:,:,n)*Wn; clutter=eta*C(:,:,n)*Wn;
    unit=max(norm(target,'fro')+norm(clutter,'fro'),realmin);
    % Separate positive target/clutter gains, QoS correction, carrier step.
    dir=(exp(tanh(z(1)))*target-exp(tanh(z(2)))*clutter)/unit ...
        +(.15+.15*tanh(z(3)))*qos/K;
    step=baseStep*exp(tanh(z(4)));
    Y(:,:,n)=Wn+sqrt(x.p.Pt/N)*step*dir;
end
end

