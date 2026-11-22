from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parent
t=pd.read_csv(root/'scnr_vs_sinr_three_methods.csv')
assert len(t)==36 and not t.duplicated(['K','gamma','kernel']).any()
print(t[['attempted','paired','AO_usable','fixedProposal_usable','learnedProposal_usable']].sum())
for key,r in t.groupby('kernel'):
    print(key,(r.AO_dB-r.fixedProposal_dB).agg(['min','max']), (r.AO_dB-r.learnedProposal_dB).agg(['min','max']))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(7.1,4.1),sharex=True,sharey=True)
styles=[('AO','AO–SDP','#262626','-','d'),('fixedProposal','Fixed proposal','#26709b','--','s'),('learnedProposal','Learned proposal','#b84920',':','o')]
for j,k in enumerate((4,8,12)):
    for i,kernel in enumerate(('estimated','perfect')):
        ax=axes[i,j]
        r=t[(t.K==k)&(t.kernel==kernel)].sort_values('gamma')
        for col,label,color,ls,m in styles:
            ax.plot(r.gamma,r[col+'_dB'],label=label,color=color,ls=ls,marker=m,mfc='none',ms=3.5,lw=1)
        ax.set_title(f'{k} users, {kernel} kernel',fontsize=8)
        ax.set_xticks([0,4,8,12,16,20])
        ax.grid(alpha=.18)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
fig.supxlabel('Required communication SINR (dB)',fontsize=9)
fig.supylabel('True sensing SCNR (dB)',fontsize=9)
fig.tight_layout(rect=(.015,.025,1,.93))
fig.savefig(root/'paper/sinr-sweep.pdf',bbox_inches='tight')
