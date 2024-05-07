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
import numba
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
quantiles = np.linspace(0, 1, num_intervals + 1)
weights = -np.log(1 - np.diff(quantiles[:-1]) / (1 - quantiles[:-2]))

# pair coalescence rates with all samples
mean_rates = np.zeros(num_intervals)
mean_breaks = np.zeros(num_intervals)
for i, f in enumerate(snakemake.input.coalrate):
    inp = pickle.load(open(f, "rb"))
    rates, breaks = np.squeeze(inp['rates']), np.squeeze(inp['breaks'])
    if i >= num_burnin:
        mean_rates += rates / (num_mcmc - num_burnin)
        mean_breaks += breaks / (num_mcmc - num_burnin)
        plt.step(breaks, rates, color='gray', alpha=0.1)
mean_rates[:-1] = weights / np.diff(mean_breaks)

plt.step(mean_breaks, mean_rates, color='black', linewidth=2)
plt.xlabel("Generations in past")
plt.ylabel("Pair coalescence rate")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig(snakemake.output.pair_coalescence_rates)
plt.clf()

# cross coalescence rates between strata
if snakemake.params.stratify is not None:
    inp = pickle.load(open(snakemake.input.crossrate[0], "rb"))
    names, rates, breaks = inp['names'], inp['rates'], inp['breaks']

    ncol = 3
    nrow = int(np.ceil(names.size / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol + 1, 4 * nrow), sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab10", names.size)

    mean_rates = rates / num_mcmc
    mean_breaks = breaks / num_mcmc
    for i, f in enumerate(snakemake.input.crossrate[1:]):
        if i + 1 >= num_burnin:
            inp = pickle.load(open(f, "rb"))
            rates, breaks = inp['rates'], inp['breaks']
            mean_rates += rates / (num_mcmc - num_burnin)
            mean_breaks += breaks / (num_mcmc - num_burnin)
            for j, p in enumerate(names):
                row, col = j // ncol, j % ncol 
                for k, q in enumerate(names):
                    axs[row, col].step(
                        breaks[j, k], 
                        rates[j, k], 
                        color=cmap(k / names.size), 
                        alpha=0.02,
                    )

    # all rates except terminal are one-to-one functions of interval width:
    indices = np.arange(names.size)
    for j, k in itertools.product(indices, indices):
        mean_rates[j, k, :-1] = weights / np.diff(mean_breaks[j, k])

    for j, p in enumerate(names):
        row, col = j // ncol, j % ncol 
        for k, q in enumerate(names):
            axs[row, col].step(
                mean_breaks[j, k], 
                mean_rates[j, k], 
                color=cmap(k / names.size), 
                label=q, 
                linewidth=2,
            )
        axs[row, col].set_title(f"With {p}", loc='center')
        axs[row, col].set_yscale('log')
        axs[row, col].set_xscale('log')
    fig.supxlabel("Generations in the past")
    fig.supylabel("Coalescence rate")
    legend = []
    for k, q in enumerate(names):
        legend.append(Line2D([0], [0], color=cmap(k / names.size), label=q, lw=2))
    fig.legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
plt.savefig(snakemake.output.cross_coalescence_rates, bbox_inches="tight")
