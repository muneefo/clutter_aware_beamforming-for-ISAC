"""Render the completed results using only the included CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm

p=Path(__file__).resolve().parent
tr=pd.read_csv(p/'outer_trace.csv').sort_values(['seed','iteration'])
s=pd.read_csv(p/'sweep_trials.csv'); bench=pd.read_csv(p/'full_run_summary.csv')
blue='#245981'; jade='#087667'; orange='#a34b22'; plum='#77538b'; black='#222222'
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':10.2,'axes.labelsize':9.6,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
                     'pdf.fonttype':42,'legend.fontsize':8.2,'axes.axisbelow':True})
def grid(ax):
    ax.grid(axis='y',lw=.45,color='#d6dadd')
    ax.tick_params(length=3,width=.65)
def save(fig,pdf,name):
    fig.savefig(p/(name+'.pdf'),bbox_inches='tight',pad_inches=.08)
    fig.savefig(p/(name+'.png'),dpi=190,bbox_inches='tight',pad_inches=.08)
    pdf.savefig(fig,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
def score_table():
    rows=[]
    for key,g in s[s.study.isin(['power','sinr'])].groupby(['study','dbm','K','gamma','method','kernel']):
        good=g[(g.status=='ok')&np.isfinite(g.trueDB)]
        mean=10*np.log10(np.mean(10**(good.trueDB/10))) if len(good) else np.nan
        rows.append([*key,len(g),len(good),mean])
    q=pd.DataFrame(rows,columns=['study','dbm','K','gamma','method','kernel','attempted','usable','meanLinearTrueSCNRdB'])
    return q
q=score_table();q.to_csv(p/'sweep_means.csv',index=False)
pilot=pd.read_csv(p/'pilot_matlab.csv')
with PdfPages(p/'objective_feasibility_sweeps.pdf') as pdf:
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.1),layout='constrained')
    ax=axes[0]
    assert not tr.duplicated(['seed','iteration']).any()
    for col,color,marker,label in [('designDB',blue,'o','Design objective'),('trueDB',jade,'s','True-kernel score')]:
        arr=tr.pivot(index='iteration',columns='seed',values=col)
        # Terminal iterates may be carried forward only after a completed early stop.
        assert arr.iloc[0].notna().all()
        arr=arr.ffill(axis=0);x=arr.index
        med=arr.median(axis=1);lo=arr.quantile(.25,axis=1);hi=arr.quantile(.75,axis=1)
        ax.fill_between(x,lo,hi,color=color,alpha=.14,linewidth=0)
        ax.plot(x,med,'-' if col=='designDB' else '--',marker=marker,mfc='white',ms=4.7,color=color,lw=1.35,label=label)
    ax.set(title='(a) Classical solver convergence',xlabel='Completed outer updates',ylabel='Absolute SCNR (dB)',xticks=np.arange(6))
    ax.legend(frameon=False,loc='lower right');grid(ax)
    ax=axes[1]; t=np.arange(4)
    specs=[('AO',black,'D'),('Physics_15',blue,'o'),('Fixed_6',orange,'s'),('Learned_6',jade,'^')]
    for j,(m,c,mark) in enumerate(specs):
        name={'AO':'AO','Physics_15':'Physics (15 LPs)','Fixed_6':'Fixed (6 LPs)','Learned_6':'Learned (6 LPs)'}[m]
        ax.plot(t+(j-1.5)*.105,pilot[m],ls='none',marker=mark,ms=5,mec=c,mfc='white' if j==2 else c,label=name)
    ax.set(title='(b) Returned beams, MATLAB pilot',xlabel='Channel draw',ylabel='True-kernel SCNR (dB)',xticks=t,xticklabels=pilot.scene.tolist())
    ax.set_ylim(13.28,13.76)
    ax.legend(loc='upper left',ncol=2,frameon=False,fontsize=7.1,columnspacing=.6,handlelength=.9)
    grid(ax);save(fig,pdf,'absolute_objective')
    fig,ax=plt.subplots(figsize=(6.05,2.9),layout='constrained')
    cond=[(4,8),(4,12),(6,8)];modes=['AO','physics','fixedProposal','learnedProposal']
    counts=np.zeros((4,3),int);den=counts.copy()
    for i,m in enumerate(modes):
        for j,(k,g) in enumerate(cond):
            z=bench[(bench['mode']==m)&(bench.K==k)&(bench.gammaDB==g)]
            assert z.feasible.nunique()==1 and z.attempted.nunique()==1
            counts[i,j]=z.feasible.iloc[0];den[i,j]=z.attempted.iloc[0]
    cm=ListedColormap(['#e9eff0','#246771'])
    ax.imshow(counts/den,aspect='auto',cmap=cm,norm=BoundaryNorm([0,.9999999,1.1],cm.N))
    for i,j in np.ndindex(counts.shape):
        ax.text(j,i,f'{counts[i,j]}/{den[i,j]}   ({counts[i,j]/den[i,j]:.1%})',
                color='white' if counts[i,j]==den[i,j] else '#253a40',ha='center',va='center')
    for y in [.5,1.5,2.5]:ax.axhline(y,color='white',lw=1.2)
    for x in [.5,1.5]:ax.axvline(x,color='white',lw=1.2)
    ax.set_yticks(np.arange(4),['AO','Physics only','Fixed proposal','Learned proposal'])
    ax.set_xticks(np.arange(3),['4 users, 8 dB','4 users, 12 dB','6 users, 8 dB'])
    ax.set_title('Feasible returned beams / attempted channel draws')
    ax.tick_params(length=0,pad=6)
    for sp in ax.spines.values():sp.set_visible(False)
    save(fig,pdf,'beam_feasibility')
    fig,ax=plt.subplots(figsize=(5.95,3.75),layout='constrained')
    sets=[('joint',jade,'o','ISAC, 6 LPs'),('radar',blue,'s','Radar only'),('heuristic',plum,'d','Heuristic'),('comm',orange,'^','Comm only')]
    for m,c,mk,label in sets:
        for ker in (['perfect'] if m=='comm' else ['perfect','estimated']):
            z=q[(q.study=='power')&(q.method==m)&(q.kernel==ker)].sort_values('dbm')
            ax.plot(z.dbm,z.meanLinearTrueSCNRdB,lw=1.4,ls='-' if ker=='perfect' else '--',marker=mk,
                    ms=4.6 if ker=='perfect' else 3.3,mfc='none' if ker=='estimated' else c,
                    color=c,label=label+('' if m=='comm' else f' ({ker})'))
    ax.set(xlabel='Transmit power (dBm)',ylabel='True-kernel SCNR (dB)',xticks=np.arange(30,51,2),
           title='Power sweep: 6 users, 10 dB communication requirement')
    ax.legend(frameon=True,ncol=1,loc='lower right',fontsize=7,framealpha=.96,labelspacing=.27,borderpad=.4)
    grid(ax);save(fig,pdf,'scnr_vs_power')
    fig,ax=plt.subplots(figsize=(5.95,3.75),layout='constrained')
    for k,c,mk in [(4,blue,'o'),(8,jade,'s'),(12,plum,'^')]:
        for ker in ['perfect','estimated']:
            z=q[(q.study=='sinr')&(q.K==k)&(q.kernel==ker)].sort_values('gamma')
            ax.plot(z.gamma,z.meanLinearTrueSCNRdB,lw=1.4,ls='-' if ker=='perfect' else '--',marker=mk,
                    ms=4.7,mfc=c if ker=='perfect' else 'white',color=c,label=f'K = {k} ({ker})')
    ax.set(xlabel='Communication SINR threshold (dB)',ylabel='True-kernel SCNR (dB)',
           xticks=np.arange(0,21,4),title='SINR sweep: 50 dBm transmit power, 6-LP joint design')
    ax.legend(frameon=False,ncol=2,fontsize=7.5,loc='lower left',columnspacing=.8);grid(ax)
    save(fig,pdf,'scnr_vs_sinr')
