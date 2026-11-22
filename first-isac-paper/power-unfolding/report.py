"""Summarize paired held-out runs and export the figures without re-running training."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

METHODS=["exact_LP","fixed_PD","learned_PD"]
LABELS=["Exact LP","Fixed steps","Learned steps"]
COLORS=["#292f34","#377297","#ad522b"]
STYLES=["-d","--s","-o"]
MEAN=lambda db: 10*np.log10(np.mean(10**(db/10)))


def paired_report(t,full=False):
    rows=[]
    time="ms" if full else "returnedMS"
    def add(g,scope):
        fixed=g[g.method=="fixed_PD"].set_index(["K","gammaDB","seed"]).sort_index()
        learn=g[g.method=="learned_PD"].set_index(["K","gammaDB","seed"]).sort_index()
        oracle=g[g.method=="exact_LP"].set_index(["K","gammaDB","seed"]).sort_index()
        assert fixed.index.equals(learn.index) and learn.index.equals(oracle.index)
        gains=learn.trueDB-fixed.trueDB
        # Draws are matched across load and SINR. Resample seed clusters.
        by_seed=gains.groupby("seed").mean().to_numpy()
        rng=np.random.default_rng(894)
        boots=rng.choice(by_seed,(5000,len(by_seed)),replace=True).mean(axis=1)
        for m,r in zip(METHODS,[oracle,fixed,learn]):
            z=dict(scope=scope,method=m,n=len(r),feasible=int(r.feasible.sum()),
                   medTrueGapDB=np.median(oracle.trueDB-r.trueDB),medTimeMS=r[time].median(),
                   p95TimeMS=r[time].quantile(.95),meanLinearTrueDB=MEAN(r.trueDB),
                   trueHits=int(((oracle.trueDB-r.trueDB<=.1) & (r.feasible==1) & (oracle.feasible==1)).sum()),
                   medSpeedupVsExact=np.median(oracle[time]/r[time]))
            if not full:
                z.update(medBankGapDB=r.bankGapDB.median(),designHits=int(r.qualityHit.sum()),
                         fallbackCount=int(r.fallback.sum()),medSolverMS=r.solverMS.median(),
                         medCertifiedMS=r.certifiedMS.median(),medianBoundDB=r.boundDB.median())
            if m=="learned_PD":
                z.update(meanLearnedMinusFixedTrueDB=gains.mean(),medianLearnedMinusFixedTrueDB=gains.median(),
                         meanGainClusterLowDB=np.quantile(boots,.025),meanGainClusterHighDB=np.quantile(boots,.975))
            rows.append(z)
    add(t,"All")
    for k,g in t.groupby("K"): add(g,f"K={k}")
    return pd.DataFrame(rows)


def render(path):
    t=pd.read_csv(path/"block_trials.csv"); h=pd.read_csv(path/"hybrid_trials.csv")
    assert len(t)==216*3 and len(h)==24*3
    assert not t.duplicated(["K","gammaDB","seed","method"]).any()
    assert not h.duplicated(["K","gammaDB","seed","method"]).any()
    assert t.feasible.eq(1).all() and h.feasible.eq(1).all()
    S=paired_report(t); H=paired_report(h,True)
    S.to_csv(path/"block_summary.csv",index=False); H.to_csv(path/"hybrid_summary.csv",index=False)
    print(S.loc[S.scope=="All"].to_string(index=False));print(H[H.scope=="All"].to_string(index=False))
    plt.rcParams.update({"font.family":"serif","font.size":10,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    with PdfPages(path/"unfolding_results.pdf") as pdf:
        fig,axes=plt.subplots(1,3,figsize=(10.8,3.4),sharey=True,layout="constrained")
        for ax,k in zip(axes,(4,8,12)):
            for m,name,col,sty in zip(METHODS,LABELS,COLORS,STYLES):
                q=t[(t.method==m)&(t.K==k)].groupby("gammaDB").trueDB.agg(MEAN)
                ax.plot(q.index,q,sty,color=col,ms=5,mfc="white",label=name,lw=1.5)
            ax.set(title=f"{k} users",xlabel="Required communication SINR (dB)",xticks=np.arange(0,21,4))
            ax.grid(axis="y",alpha=.24);axes[0].set_ylabel("True-kernel SCNR (dB)")
        axes[-1].legend(loc="best",fontsize=8,frameon=False)
        fig.suptitle("First power allocation with a shared, fixed receive filter",fontsize=13)
        pdf.savefig(fig);fig.savefig(path/"block_sweep.png",dpi=155);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(8.8,3.55),layout="constrained")
        ax=axes[0]
        for i,(m,name,col) in enumerate(zip(METHODS,LABELS,COLORS)):
            r=h[h.method==m].groupby("K").trueDB.agg(MEAN)
            ax.plot(np.arange(3)+.08*(i-1),r,"o",color=col,ms=5,mfc="white" if i==1 else col,label=name)
        ax.set(xticks=range(3),xticklabels=[4,8,12],xlabel="Number of users",ylabel="True-kernel SCNR (dB)",title="Returned beam designs")
        ax.legend(fontsize=8,frameon=False);ax.grid(axis="y",alpha=.24)
        ax=axes[1]
        for i,(m,name,col) in enumerate(zip(METHODS,LABELS,COLORS)):
            r=h[h.method==m].groupby("K").ms.median()
            ax.bar(np.arange(3)+.23*(i-1),r,width=.21,color=col,label=name)
        ax.set(xticks=range(3),xticklabels=[4,8,12],xlabel="Number of users",ylabel="Median CPU wall time (ms)",title="Full solver time")
        ax.grid(axis="y",alpha=.24); ax.set_axisbelow(True)
        fig.suptitle("Six allocations at 8 dB SINR: only the first allocator varies",fontsize=13)
        pdf.savefig(fig);fig.savefig(path/"full_solver.png",dpi=155);plt.close(fig)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("path",type=Path,nargs="?",default=Path(__file__).parent/"results")
    render(ap.parse_args().path)
