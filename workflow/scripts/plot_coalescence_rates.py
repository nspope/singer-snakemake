"""
Plot pair coalescence rates within time intervals from an ARG.

Averaging rates is a little unintuitive, because we've calculated rates within
fixed quantiles. Because the rates are a one-to-one function of the epoch
widths for the interior quantiles, we average the epoch widths and then
recalculate rates. The rate in the terminal bin may be averaged directly.

Part of https://github.com/nspope/singer-snakemake.
"""

import sys
import pickle
import msprime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
import tskit
from datetime import datetime

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

num_intervals = snakemake.params.coalrate_epochs
num_mcmc = snakemake.params.mcmc_samples
num_burnin = snakemake.params.mcmc_burnin

ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
windows = ratemap.position

# TODO: something off here why is mean lower than it should be

# pair coalescence rates with all samples
reps = 0
mean_pdf = np.zeros(num_intervals)
for i, f in enumerate(snakemake.input.coalrate):
    inp = pickle.load(open(f, "rb"))
    pdf, breaks = np.squeeze(inp['pdf']), np.squeeze(inp['breaks'])
    if i >= num_burnin:
        mean_pdf += pdf
        reps += 1
        plt.step(breaks, pdf, color='gray', alpha=0.1)
mean_pdf /= reps

plt.step(breaks, mean_pdf, color='black', linewidth=1.5)
plt.xlabel("Generations in past")
plt.ylabel("Proportion coalescing pairs")
plt.xscale("log")
plt.tight_layout()
plt.savefig(snakemake.output.pair_coalescence_pdf)
plt.clf()

# cross coalescence rates between strata
if snakemake.params.stratify is not None:
    inp = pickle.load(open(snakemake.input.crossrate[0], "rb"))
    names, pdf, breaks = inp['names'], inp['pdf'], inp['breaks']

    ncol = 3
    nrow = int(np.ceil(names.size / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol + 1, 4 * nrow), sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab10", names.size)

    reps = 0
    mean_pdf = np.zeros_like(pdf)
    for i, f in enumerate(snakemake.input.crossrate[1:]):
        if i >= num_burnin:
            inp = pickle.load(open(f, "rb"))
            pdf = inp['pdf']
            mean_pdf += pdf
            reps += 1
            for j, p in enumerate(names):
                row, col = j // ncol, j % ncol 
                for k, q in enumerate(names):
                    axs[row, col].step(
                        breaks[j, k], 
                        pdf[j, k], 
                        color=cmap(k / names.size), 
                        alpha=0.02,
                    )
    mean_pdf /= reps

    for j, p in enumerate(names):
        row, col = j // ncol, j % ncol 
        for k, q in enumerate(names):
            axs[row, col].step(
                breaks[j, k], 
                mean_pdf[j, k], 
                color=cmap(k / names.size), 
                label=q, 
                linewidth=2,
            )
        axs[row, col].set_title(f"With {p}", loc='center')
        axs[row, col].set_xscale('log')
    fig.supxlabel("Generations in the past")
    fig.supylabel("Proportion coalescencing pairs")
    legend = []
    for k, q in enumerate(names):
        legend.append(Line2D([0], [0], color=cmap(k / names.size), label=q, lw=2))
    fig.legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
plt.savefig(snakemake.output.cross_coalescence_pdf, bbox_inches="tight")
