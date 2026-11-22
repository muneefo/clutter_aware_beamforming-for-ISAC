"""Recreate the paired-budget figure from the saved numerical results."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot(source, target):
    by_seed = {}
    with source.open(newline='') as stream:
        for row in csv.DictReader(stream):
            assert row['feasible'] == '1', 'This figure requires paired feasible results.'
            by_seed.setdefault(int(row['seed']), {})[int(row['budget'])] = float(row['trueDB'])
    gain = np.sort([r[15] - r[6] for r in by_seed.values()])
    prob = np.arange(1, gain.size + 1) / gain.size
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.45, 1.91), layout='constrained')
    left = min(gain[0], 0) - .02
    right = max(gain[-1], 0) + .02
    ax.step(np.r_[left, gain, right], np.r_[0, prob, 1], where='post',
            color='#244963', linewidth=1.55)
    ax.axvline(0, color='.3', linestyle=':', linewidth=.85)
    ax.set(xlim=(left, right), ylim=(-.025, 1.03), xlabel='True SCNR gain, 15 vs. 6 LP solves (dB)',
           ylabel='Cumulative fraction')
    ax.set_yticks(np.arange(0, 1.01, .25))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='.88', linewidth=.45)
    fig.savefig(target)
    plt.close(fig)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('target', type=Path)
    args = p.parse_args()
    plot(args.source, args.target)
