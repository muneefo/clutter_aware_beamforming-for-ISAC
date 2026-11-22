"""Generate the manuscript's vector figures and paired numerical summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

out=Path("../paper")
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.spines.top':False,
                     'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42})
colors=['#19496a','#b75e28','#008679','#873579','#555555']
pdf=PdfPages('../results_figures.pdf')
def save(fig,name):
    fig.tight_layout()
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    pdf.savefig(fig,bbox_inches='tight')
    plt.close(fig)
def axes(n=2,height=2.75):
    return plt.subplots(1,n,figsize=(7.1,height),layout=None)
def decorate(ax,x,y):
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(axis='y',alpha=.22)
summary=pd.read_csv('mc-summary.csv')
pilot=pd.read_csv('pilot_scnr.csv')
fig,ax=axes()
for i,(col,label) in enumerate([('AO','AO–SDP'),('Fixed_6','Fixed proposal'),('Learned_6','Learned proposal')]):
    ax[0].plot(pilot.scene,pilot[col],['-D','--s',':o'][i],color=colors[i],label=label,
               markerfacecolor='none',ms=5)
ax[0].set_xticks(pilot.scene); ax[0].legend(fontsize=7); ax[0].set_title('(a) Four paired pilot channels')
decorate(ax[0],'Scene index','True SCNR (dB)')
for i,((k,g),r) in enumerate(summary.groupby(['K','gammaDB'])):
    f=r[r['mode']=='fixedProposal'].sort_values('outer')
    ax[1].plot(f.medianSeconds,f.hits/f.validAO,marker=['o','s','^'][i],color=colors[i],label=f'{k} users, {g} dB')
    ao=r[r['mode']=='AO'].iloc[0]
    ax[1].scatter(ao.medianSeconds,1,marker=['o','s','^'][i],facecolors='none',edgecolors=colors[i])
ax[1].set_xscale('log'); ax[1].set_ylim(-.04,1.1)
ax[1].legend(fontsize=6,loc='center right'); ax[1].set_title('(b) Fixed proposal vs. AO (open symbols)')
decorate(ax[1],'Median design time (s)','Fraction within 0.1 dB of AO')
save(fig,'ao-anchor')
df=pd.concat([pd.read_csv('trials.csv'),pd.read_csv('controls.csv')],ignore_index=True)
assert not df.duplicated(['K','gamma','seed','method']).any() and len(df)==576
agg=df.groupby(['K','gamma','method']).agg(attempted=('seed','size'),feasible=('feasible','sum'),
                                            medianSCNR=('scnr','median')).reset_index()
agg.to_csv('summary.csv',index=False)
wide=df.pivot(index=['seed','K','gamma'],columns='method',values='scnr')
assert np.all(wide.Absent>=wide.Blind-1e-9)
assert np.allclose(10*np.log10(1+df[df.method=='Blind'].sort_values(['seed','K','gamma']).coldToAdditive),wide.Absent-wide.Blind)
rng=np.random.default_rng(5931)
rows=[]
for m in ['Blind','RX only','Absent','Thermal only','ZF','RZF','MRT']:
    gain=(wide.Estimated-wide[m]).unstack(['K','gamma']).to_numpy()
    ids=rng.integers(len(gain),size=(4000,len(gain)))
    boot=np.nanmedian(gain[ids],axis=(1,2))
    lo,hi=np.quantile(boot,[.025,.975])
    rows.append(dict(comparator=m,paired=np.isfinite(gain).sum(),median=np.nanmedian(gain),low=lo,high=hi))
pd.DataFrame(rows).to_csv('paired.csv',index=False)
fig,ax=axes(2,3.0)
for i,(name,label) in enumerate([('Estimated','Joint aware'),('RX only','Aware receiver only'),('Blind','Blind Tx/Rx'),('Absent','Cold clutter absent'),('Thermal only','Thermal noise only')]):
    for j,k in enumerate([4,8]):
        r=agg[(agg.K==k)&(agg.method==name)]
        ax[j].plot(r.gamma,r.medianSCNR,color=colors[i],marker=['D','o','s','^','v'][i],
                   ls=['-','--',':','-.','--'][i],label=label,markerfacecolor='none')
for j,k in enumerate([4,8]):
    ax[j].set_title(f'{k} users');ax[j].set_xticks([0,8,16])
    decorate(ax[j],'Required user SINR (dB)','Output sensing ratio (dB)')
ax[1].legend(fontsize=6,loc='center left')
save(fig,'clutter-ablation')
fig,ax=axes()
for i,name in enumerate(['Estimated','MRT','ZF','RZF']):
    for j,k in enumerate([4,8]):
        r=agg[(agg.K==k)&(agg.method==name)]
        ax[j].plot(r.gamma,r.medianSCNR,ls=['-','--','-.',':'][i],marker=['D','x','s','o'][i],
                   color=colors[i],ms=[4,6,5,7][i],markerfacecolor='none',label='Fixed proposal' if i==0 else name)
for j,k in enumerate([4,8]):
    ax[j].set_title(f'{k} users');ax[j].set_xticks([0,8,16]);ax[j].set_ylim(6,14.2)
    decorate(ax[j],'Required user SINR (dB)','Feasible true SCNR (dB)')
ax[0].legend(fontsize=7,ncol=2)
save(fig,'precoder-comparison')
fig,ax=axes()
pts=pd.read_csv('patterns.csv').groupby(['method','angle']).fraction.mean()
for i,name in enumerate(['Estimated','Blind','ZF']):
    r=pts.loc[name]
    ax[0].plot(r.index,10*np.log10(r),color=colors[i],ls=['-','--',':'][i],label=name)
ax[0].axvline(-10,color='gray',lw=.8,ls=':'); ax[0].legend(fontsize=7)
ax[0].set_title('(a) Transmit illumination')
decorate(ax[0],'Angle (degrees)','Power relative to total (dB)')
for i,name in enumerate(['Blind','RX only','ZF']):
    difference=np.sort((wide.Estimated-wide[name]).dropna())
    ax[1].step(difference,np.arange(1,len(difference)+1)/len(difference),where='post',color=colors[i],label=name)
ax[1].axvline(0,color='gray',lw=.8,ls=':'); ax[1].legend(fontsize=7,loc='lower right');ax[1].set_title('(b) Paired gain of joint design')
decorate(ax[1],'Joint aware minus comparator (dB)','Empirical cumulative fraction')
save(fig,'physical-mechanism')
pdf.close()
