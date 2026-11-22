from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results')
meta=json.loads((R/'metadata.json').read_text());val=json.loads((R/'validation.json').read_text());tim=json.loads((R/'timings.json').read_text());paired=json.loads((R/'paired.json').read_text())
rows=list(csv.DictReader((R/'summary.csv').open()))
groups=[('Initialization',['ZF initialization']),('Fixed steps (8)',['Fixed PGA (8)']),('Learn before',[f'before_{s}' for s in [17,23,31]]),('Learn inside',[f'inside_{s}' for s in [17,23,31]]),('Before + inside',[f'combined_{s}' for s in [17,23,31]]),('Backtracking (80)',['Backtracking PGA (80)'])]
def path(dataset,name):return R/(dataset+'_'+name.replace(' ','_').replace('(','').replace(')','')+'.npz')
summary=[]
for label,names in groups:
 rr=[r for r in rows if r['dataset']=='test' and r['method'] in names]
 ood=[r for r in rows if r['dataset']=='clutter_shift' and r['method'] in names]
 summary.append(dict(method=label,feasible_percent=float(np.mean([float(r['feasible_percent']) for r in rr])),mean_scnr_db_feasible=float(np.mean([float(r['scnr_db_mean_feasible']) for r in rr])),median_ms=float(np.mean([tim[n]['median_ms'] for n in names])),p95_ms=float(np.mean([tim[n]['p95_ms'] for n in names])),ood_feasible_percent=float(np.mean([float(r['feasible_percent']) for r in ood]))))
(R/'aggregate.json').write_text(json.dumps(summary,indent=2))
# Paired bootstrap resamples geometries, holding the three trained models fixed.
rng=np.random.default_rng(991);cis={}
base=np.load(path('test','Fixed PGA (8)'))
for mode in ['before','inside','combined']:
 outs=[np.load(path('test',f'{mode}_{s}')) for s in [17,23,31]]
 common=base['feasible'].copy()
 for o in outs:common&=o['feasible']
 gains=np.mean([10*np.log10(o['scnr'][common]/base['scnr'][common]) for o in outs],0)
 boot=np.mean(gains[rng.integers(0,len(gains),(3000,len(gains)))],1)
 cis[mode]=dict(mean_db=float(gains.mean()),low=float(np.percentile(boot,2.5)),high=float(np.percentile(boot,97.5)),n_common=int(common.sum()))
(R/'paired_bootstrap.json').write_text(json.dumps(cis,indent=2))
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'mathtext.fontset':'stix'})
fig,axs=plt.subplots(2,2,figsize=(10.3,7.4),layout='constrained');colors=['#777777','#222222','#c17626','#25688d','#16775e','#84629d']
ax=axs[0,0]
for i,s in enumerate(summary):
 ax.plot(s['feasible_percent'],i,'o',color=colors[i]);ax.text(s['feasible_percent']+.6,i,f"{s['feasible_percent']:.1f}%",va='center',fontsize=8)
ax.set_yticks(range(6),[g[0] for g in groups]);ax.invert_yaxis();ax.set_xlim(75,92);ax.set_xlabel('Nominal feasible outputs (%)');ax.set_title('(a) Coverage: all 128 test geometries',loc='left');ax.grid(axis='x',alpha=.2)
ax=axs[0,1]
for i,(mode,label) in enumerate([('before','Before'),('inside','Inside'),('combined','Before + inside')]):
 c=cis[mode];ax.errorbar(c['mean_db'],i,xerr=[[c['mean_db']-c['low']],[c['high']-c['mean_db']]],fmt='o',color=colors[i+2],capsize=3)
ax.axvline(0,color='gray',ls='--',lw=.8);ax.set_yticks(range(3),['Before','Inside','Before + inside']);ax.invert_yaxis();ax.set_xlabel('Paired SCNR gain over fixed steps (dB)');ax.set_title('(b) Shared feasible cases; 95% bootstrap CI',loc='left');ax.grid(axis='x',alpha=.2)
ax=axs[1,0]
for i,s in enumerate(summary):
 ax.plot([s['median_ms'],s['p95_ms']],[i,i],color=colors[i],lw=2);ax.plot(s['median_ms'],i,'o',color=colors[i]);ax.plot(s['p95_ms'],i,'|',color=colors[i],ms=8)
ax.set_xscale('log');ax.set_yticks(range(6),[g[0] for g in groups]);ax.invert_yaxis();ax.set_xlabel('Solver runtime (ms; dot: median, tick: p95)');ax.set_title('(c) Batch 1; checking and repair included',loc='left');ax.grid(axis='x',alpha=.2)
ax=axs[1,1]
common=base['feasible'].copy()
for label,names in groups:
 for n in names:common&=np.load(path('test',n))['feasible']
for i in [1,3,4,5]:
 label,names=groups[i];db=np.mean([10*np.log10(np.load(path('test',n))['scnr'][common]) for n in names],0);x=np.sort(db)
 ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label,color=colors[i],lw=1.3)
ax.set_xlabel('SCNR (dB; seed-mean per geometry)');ax.set_ylabel('Empirical CDF');ax.set_title(f'(d) Common feasible subset: {common.sum()} cases',loc='left');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
fig.savefig(R/'simulation-summary.pdf',bbox_inches='tight');fig.savefig(R/'simulation-summary.png',dpi=170,bbox_inches='tight');plt.close(fig)
# Numerical system geometry for the first held-out case.
d=np.load(R/'test.npz');p=d['points'][0];cfg=meta['config'];fig,ax=plt.subplots(figsize=(6.4,4.4),layout='constrained')
ax.scatter(np.zeros(cfg['N']),np.linspace(-.175,.175,cfg['N']),marker='s',s=20,c='black',label='BS elements')
ax.scatter(np.full(cfg['M'],.65),np.linspace(.15,.7,cfg['M']),marker='s',s=15,c='#25688d',label='RIS elements')
ax.scatter(p[:2,0],p[:2,1],marker='o',s=40,c='#16775e',label='Users');ax.scatter(p[2,0],p[2,1],marker='*',s=130,c='#9c3434',label='Target');ax.scatter(p[3:,0],p[3:,1],marker='x',s=45,c='#c17626',label='Clutter')
ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left');ax.set_title('First held-out synthetic geometry; dimensions to scale',fontsize=10)
fig.savefig(R/'simulated-geometry.pdf',bbox_inches='tight');fig.savefig(R/'simulated-geometry.png',dpi=170,bbox_inches='tight');plt.close(fig)
# LaTeX results generated only from saved measured outputs.
text=r'''\subsection{Checks actually executed}
Analytical gradients were compared with central finite differences on all real and imaginary beamforming entries and all RIS phases for five random states. The step was $10^{-6}$ and the optimal receiver was recomputed at perturbed points. The maximum relative gradient error was $%.3g$. Independent generalized-eigenvalue receiver verification gave maximum relative error $%.3g$. A separate 128-instance zero-phase initialization test accepted %d cases; failures were retained. The accepted power limit and phase modulus were satisfied to floating-point tolerance.

\subsection{Held-out simulation results}
Table~\ref{tab:results} uses 128 held-out geometries. Learned-method entries average the three training seeds; conditional SCNR averages use each model's own feasible subset and must not be interpreted as paired comparisons. ``Feasible'' refers to nominal communication QoS and hardware constraints, not a sensing detection guarantee. No algorithm repairs failed initializations in this prototype.

\begin{table}[!htbp]\centering\small
\begin{tabular}{lrrrr}\toprule
Method & Feasible (\%%) & Mean SCNR (dB) & Median (ms) & p95 (ms)\\\midrule
'''%(max(val['gradient_relative_errors']),max(val['receiver_relative_errors']),val['zf_feasible_count'])
for s in summary:text+=f"{s['method']} & {s['feasible_percent']:.2f} & {s['mean_scnr_db_feasible']:.2f} & {s['median_ms']:.2f} & {s['p95_ms']:.2f} \\\\\n"
text+=r'''\bottomrule\end{tabular}
\caption{Actual synthetic test results and batch-one solver timings. Learned timing entries are means of per-seed medians/p95 values, not pooled quantiles.}\label{tab:results}\end{table}

Runtime is measured with \texttt{perf\_counter} on this shared Linux CPU environment using NumPy complex128 and one OpenBLAS thread. Each method uses three repetitions over 32 batch-one inputs. Timing includes initialization, receiver solves, gradient steps, power repair, candidate evaluation, and final checks. It excludes offline learning and synthetic channel/feature construction. It also excludes sensing acquisition, channel estimation, RIS configuration, and queueing, none of which is simulated. Consequently these numbers are solver timings, not end-to-end deployment latency. Median synthetic input preparation took %.3f ms; this is not an estimator timing. Offline training of all nine learned models took %.2f s.

On geometries feasible for both the learned and fixed-step methods, the seed-averaged paired SCNR gain was %.3f dB for inside-only learning, %.3f dB for combined before/inside learning, and %.3f dB for before-only learning. The corresponding geometry-bootstrap 95\%% intervals were [%.3f, %.3f], [%.3f, %.3f], and [%.3f, %.3f] dB. These intervals hold the three trained models fixed and do not capture all training-distribution or model uncertainty.

The combined architecture did not consistently exceed inside-only learning. Before-only learning traded a small feasibility change for lower paired sensing quality. There is no demonstrated runtime reduction relative to the equally shallow eight-step baseline; the 80-step reference spends more time and attains higher average sensing quality. The results support a modest learned-step benefit in this pilot, not a general L2O superiority claim.

\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth]{results/simulation-summary.pdf}
\caption{Measured results from this prototype. Coverage retains failed cases; paired gains and the CDF use shared-feasible subsets. Confidence intervals resample geometries with trained models held fixed.}\end{figure}

\subsection{Clutter shift and dimension diagnostic}
The shifted test multiplies clutter RCS by three and concentrates clutter around the target; it also uses new user and target geometries. It is therefore not a paired intervention on the same scenes. Inside-only learning retained an average paired improvement of %.3f dB over fixed steps on their shared feasible cases; combined learning gave %.3f dB. Neither result establishes robust feasibility under unknown channel or clutter errors.
'''%(meta['common_synthetic_input_preparation_median_ms'],meta['training_seconds'],cis['inside']['mean_db'],cis['combined']['mean_db'],cis['before']['mean_db'],cis['inside']['low'],cis['inside']['high'],cis['combined']['low'],cis['combined']['high'],cis['before']['low'],cis['before']['high'],np.mean(paired['clutter_shift_inside']['mean_gain_db_per_seed']),np.mean(paired['clutter_shift_combined']['mean_gain_db_per_seed']))
if (R/'dimension_diagnostic.json').exists():
 dim=json.loads((R/'dimension_diagnostic.json').read_text());cov=json.loads((R/'covariance_check.json').read_text())
 text+=r'''An additional frozen-seed-17 diagnostic changes $(N,M)$ from $(6,12)$ to $(12,24)$ and $(24,48)$, retaining $K=2$ and fixed physical apertures. It uses 32 fresh instances at each size, with no retraining. It tests execution and limited dimension transfer only; it does not validate arbitrary-user scalability or array hardware physics.
\begin{center}\small\begin{tabular}{rrlrr}\toprule
$N$ & $M$ & Method & Feasible / 32 & Median (ms)\\\midrule
'''
 for r in dim:text+=f"{r['N']} & {r['M']} & {r['method']} & {r['feasible']} & {r['median_ms']:.2f} \\\\\n"
 text+=r'\bottomrule\end{tabular}\end{center}'+'\n'
 text+=f"An independent covariance check with {cov['samples']:,} random symbol/scatterer draws produced SCNR relative error {cov['relative_error']:.4f} against the analytical expression. This checks the assumed ensemble covariance model, not its validity for measured clutter.\n"
text+=r'''\begin{figure}[!htbp]\centering
\includegraphics[width=.75\linewidth]{results/simulated-geometry.pdf}
\caption{Coordinates of the first held-out case used by the simulator. This figure is generated from the saved test dataset, not an illustrative performance claim.}\end{figure}
'''
(R/'results-section.tex').write_text(text)
print(json.dumps(summary,indent=2));print(json.dumps(cis,indent=2))
