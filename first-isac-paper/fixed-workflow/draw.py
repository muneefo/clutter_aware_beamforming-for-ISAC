from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = Path(__file__).parent
plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42})
fig = plt.figure(figsize=(11,6.6), facecolor="white")
ax = fig.add_axes([0,0,1,1], xlim=(0,11), ylim=(0,6.6))
ax.axis("off")
ink, muted, teal = "#173145", "#476171", "#00675e"

def label(x,y,s,size=11,color=ink,weight="normal"):
    ax.text(x,y,s,va="center",fontsize=size,color=color,weight=weight)

def card(x,y,w,h,title,detail,fill="white",edge="#9bafb9"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.05",
                                facecolor=fill,edgecolor=edge,linewidth=.9))
    label(x+.18,y+h-.20,title,11.4,ink,"bold")
    ax.text(x+.18,y+h-.37,detail,va="top",fontsize=9.3,color=muted,linespacing=1.32)

def arrow(x,y,a,b):
    ax.add_patch(FancyArrowPatch((x,y),(a,b),arrowstyle="-|>",mutation_scale=12,
                                 color=muted,lw=1.05,shrinkA=0,shrinkB=0))

label(.45,6.21,"Fixed-proposal ISAC beamforming",22,weight="bold")
label(.46,5.86,"Transmit directions, power allocation, and receive filtering",11.2,muted)
ax.plot([.45,10.54],[5.62,5.62],color=ink,lw=1)
for x,y,w,h,t,s in [
    (.92,4.70,5.05,.68,"1  Initialize feasible beams","Zero-force user interference. Put leftover power into sensing."),
    (.92,3.81,5.05,.68,"2  Update the receive filters","Use MVDR with the current beams and the supplied clutter model."),
    (.92,2.92,5.05,.68,"3  Propose user-beam directions","Combine target, clutter, and SINR terms with fixed weights."),
    (.92,1.94,5.05,.77,"4  Allocate power and recover beams","Solve one LP over all tones. Recover physical transmit beams."),
    (.92,.87,5.05,.86,"5  Check the recovered beams","Accept if SINR and power pass and design SCNR is preserved\nwithin tolerance. Otherwise keep the previous beams."),
]:
    card(x,y,w,h,t,s,"#eaf5f2" if y in (2.92,1.94) else "white",teal if y in (2.92,1.94) else "#9bafb9")
for y,b in [(4.70,4.49),(3.81,3.60),(2.92,2.71),(1.94,1.73)]:
    arrow(3.43,y,3.43,b)

card(6.42,3.87,4.13,1.49,"What goes into the proposal",
     "+  Increase target return\n−  Suppress clutter at the current receiver\n+  Correct user SINR")
card(6.42,2.32,4.13,1.25,"What the LP decides",
     "Assign power to each candidate. A direction\ncan receive zero power. SINR requirements\nand total power constrain the allocation.","#eaf5f2",teal)
card(6.42,.87,4.13,1.15,"What the LP sends back",
     "Dual prices identify new directions worth\ntesting in a subsequent allocation.")
ax.plot([5.99,6.40],[2.33,2.58],color=muted,lw=.85,ls=":")
ax.plot([5.99,6.40],[3.28,4.10],color=muted,lw=.85,ls=":")
ax.plot([.89,.53,.53],[1.21,1.21,4.16],color=muted,lw=1)
arrow(.53,4.16,.90,4.16)
ax.text(.31,2.68,"Refresh MVDR after transmit updates",rotation=90,ha="center",va="center",fontsize=9,color=muted)
label(.92,.47,"Six-LP setting",10,teal,"bold")
label(2.12,.47,"Up to 3 outer updates × 2 transmit LPs. Return the final beams and refreshed filters.",10,muted)
for ext in ("png","pdf"):
    fig.savefig(out/f"fixed_proposal_workflow.{ext}",dpi=180)
