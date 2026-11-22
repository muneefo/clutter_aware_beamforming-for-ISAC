function out = fixed_array_joint(mode,seed,trials)
% Joint-confidence solver for the 24-Tx, 16-Rx, two-reflector ISAC model.
% fixed_array_joint('verify'); out=fixed_array_joint('simulate',42,128);
if nargin<1, mode='simulate'; end
if nargin<2, seed=42; end
if nargin<3, trials=128; end
here=fileparts(mfilename('fullpath'));
model=jsondecode(fileread(fullfile(here,'joint_model.json')));
if strcmpi(mode,'verify')
    f=load(fullfile(here,'joint_fixture.mat'));
    for i=1:size(f.W,3)
        V=vertices(f.C(:,:,i),f.b(:,i));
        [val,~,W]=design(f.t(:,:,i),f.r(:,:,i),f.h(:,:,i),V,f.hat(:,i),model.theta,model.steps);
        assert(abs(val-f.bound(i))<5e-7*max(1,val),'SCNR fixture mismatch');
        assert(norm(W-f.W(:,:,i),'fro')<1e-5,'Precoder fixture mismatch');
    end
    assert(isempty(vertices([eye(2);-eye(2)],[-1;-1;0;0])),'Empty set test failed');
    out=true; disp('Joint solver fixtures passed.'); return
end
assert(strcmpi(mode,'simulate'),'Use simulate or verify.');
validateattributes(trials,{'numeric'},{'scalar','integer','positive'});
saved=rng; clean=onCleanup(@()rng(saved)); %#ok<NASGU>
rng(seed,'twister');
base=[.035 .73;.085 .70;-.12 .82;-.19 .66;.20 .84];
geom=[1;zeros(5,1)]; fixed=.12*ones(1,3);
coefs={geom,geom,model.theta,model.theta}; schedules={fixed,model.steps,fixed,model.steps};
values=nan(trials,4,5); valid=false(trials,1); covered=false(trials,1);
for i=1:trials
    pos=base+.022*(2*rand(size(base))-1); truth=exp(-.8+1.8*rand(2,1));
    alpha=(randn(96,6,2)+1i*randn(96,6,2))/sqrt(2);
    noise=(randn(96,6)+1i*randn(96,6))/sqrt(2);
    tx=steer(pos,24,-.008); rx=steer(pos,16,.008);
    t=tx(:,1:3); r=rx(:,1:3); h=tx(:,4:5).';
    G=zeros(6,2);
    for k=0:5
        f=conj(t(:,mod(k,3)+1)); f=f/norm(f);
        u=r(:,mod(k+floor(k/3),3)+1); u=u/norm(u);
        G(k+1,:)=(u'*r(:,2:3)).*(t(:,2:3).'*f).';
    end
    echo=sqrt(2e-11)*noise;
    for j=1:2, echo=echo+alpha(:,:,j).*(sqrt(truth(j))*G(:,j).'); end
    H=abs(G).^2/2e-11; energy=mean(abs(echo).^2,1).'/2e-11;
    assert(rank(H)==2,'Probe power matrix is unidentifiable');
    candidates=[zeros(2,1),diag(max(0,(H.'*(energy-1))./sum(H.^2,1).'))];
    x=H\(energy-1);
    if all(x>=0), candidates(:,end+1)=x; end
    [~,k]=min(sum((H*candidates-(energy-1)).^2,1)); hat=candidates(:,k);
    tail=.05/12;
    C=[H;-H;-eye(2)]; b=[96*energy/gammaincinv(tail,96)-1;1-96*energy/gammaincinv(1-tail,96);0;0];
    V=vertices(C,b);
    if isempty(V), continue; end
    valid(i)=true; covered(i)=all(C*truth<=b+1e-9);
    for k=1:4
        [val,u,W]=design(t,r,h,V,hat,coefs{k},schedules{k});
        g=sum(abs(t.'*W).^2,2).*abs(r'*u).^2/2e-11;
        actual=g(1)/(1+truth.'*g(2:end)); sinr=quality(h,W);
        assert(~covered(i)||val<=actual+1e-8,'Coverage-event bound failed.');
        assert(min(sinr)>=2-1e-9 && norm(W,'fro')^2<=1+1e-9,'Infeasible design.');
        values(i,k,:)=[val actual sinr.' norm(W,'fro')^2];
    end
end
out.methods={'geometry_fixed','geometry_matched','learned_init_fixed','combined'};
out.columns={'joint_bound','true_scnr','user1_sinr','user2_sinr','power_W'};
out.metrics=values; out.valid=valid; out.covered=covered; out.model=model;
fprintf('%d of %d scenes returned designs. SCNR values in out.metrics are linear.\n',nnz(valid),trials);
end

function A=steer(pos,n,offset)
x=((0:n-1).'-(n-1)/2)*.005+offset;
d=hypot(x-pos(:,1).',pos(:,2).'); A=.01./(4*pi*d).*exp(-2i*pi*d/.01);
end

function V=vertices(C,b)
V=zeros(0,2);
for i=1:length(b)-1
    for j=i+1:length(b)
        A=C([i,j],:);
        if cond(A)>1e10, continue; end
        v=A\b([i,j]);
        if all(C*v<=b+1e-9), V(end+1,:)=v.'; end %#ok<AGROW>
    end
end
if isempty(V), return; end
V=unique(V,'rows'); d=V-mean(V,1); [~,k]=sort(atan2(d(:,2),d(:,1))); V=V(k,:);
end

function s=quality(h,W)
p=abs(h*W).^2; s=diag(p)./(6e-7+sum(p,2)-diag(p));
end

function [val,u,W]=design(t,r,h,V,hat,theta,steps)
Z=h'/(h*h'); W0=(Z./vecnorm(Z))/sqrt(2);
v=conj(t(:,1))-Z*(h*conj(t(:,1)));
T=(v/norm(v))*exp(1i*angle(t(:,1).'*W0))/sqrt(2);
features=[1;log1p(hat);log(quality(h,W0)/2);100*norm(t(:,1))];
f=max(0,min(.94,features.'*theta(:)));
W=sqrt(1-f)*W0+sqrt(f)*T;
W=project(W.*exp(-1i*angle(diag(h*W))).',h);
[val,u,g]=receiver(t,r,W,V);
for eta=steps(:).'
    if norm(g,'fro')<1e-15, continue; end
    for j=0:8
        Y=project(W+eta*2^(-j)*g/norm(g,'fro'),h);
        [nv,nu,ng]=receiver(t,r,Y,V);
        if nv>=val-1e-12, W=Y; val=nv; u=nu; g=ng; break; end
    end
end
end

function [v,d]=response(q,r,illum)
R=eye(size(r,1))+(r(:,2:3).*(q(:).*illum(2:3)/2e-11).')*r(:,2:3)';
v=R\r(:,1); d=abs(r(:,2:3)'*v).^2.*illum(2:3)/2e-11;
end

function s=slope(a,e,x,r,illum)
[~,d]=response(a+x*e,r,illum); s=e*d;
end

function ok=cert(q,V,r,illum)
[v,d]=response(q,r,illum);
ok=max((V-q)*d)<=1e-11*max([norm(v)^2,abs(q*d),1e-30]);
end

function [val,u,grad]=receiver(t,r,W,V)
illum=sum(abs(t.'*W).^2,2); q=[];
for k=1:size(V,1)
    if cert(V(k,:),V,r,illum), q=V(k,:); break; end
end
if isempty(q)
    for k=1:size(V,1)
        a=V(k,:); e=V(mod(k,size(V,1))+1,:)-a;
        fun=@(x)slope(a,e,x,r,illum);
        if fun(0)>0 && fun(1)<0
            z=a+fzero(fun,[0,1])*e;
            if cert(z,V,r,illum), q=z; break; end
        end
    end
end
assert(~isempty(q),'No certified receiver saddle point');
u=response(q,r,illum); u=u/norm(u);
weights=abs(r'*u).^2/2e-11; den=1+max(V*(weights(2:3).*illum(2:3)));
val=weights(1)*illum(1)/den;
c=[weights(1);-val*q(:).*weights(2:3)]/den;
grad=(conj(t).*c.')*(t.'*W);
end

function X=project(raw,h)
n=vecnorm(h,2,2); H=h./n; sigma=sqrt(6e-7)./n; root=sqrt(2);
corr=zeros([size(raw),3],'like',raw); X=raw;
for iter=1:5000
    movement=0;
    for i=1:3
        Y=X+corr(:,:,i);
        if i==3
            next=Y/max(1,norm(Y,'fro'));
        else
            a=H(i,:); y=a*Y; dest=y; j=3-i; s=real(y(i)); mag=abs(y(j));
            if s>=root*hypot(mag,sigma(i))
                dest(i)=s;
            else
                fun=@(k) k*root*(root*hypot(k*mag,sigma(i))-s)-(1-k)*hypot(k*mag,sigma(i));
                k=fzero(fun,[0,1]); dest(j)=k*y(j); dest(i)=root*hypot(k*mag,sigma(i));
            end
            next=Y+a'*(dest-y);
        end
        movement=max(movement,norm(next-X,'fro')); corr(:,:,i)=Y-next; X=next;
    end
    if movement<2e-11 && min(quality(h,X))>=2-1e-9 && norm(X,'fro')^2<=1+1e-9, return; end
end
error('Projection did not converge');
end
