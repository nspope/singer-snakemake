"""
Plot pair coalescence rates and time distributions from an ARG.

Part of https://github.com/nspope/singer-snakemake.
"""

import sys
import pickle
import msprime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
import warnings
import tskit
import os
from datetime import datetime

warnings.simplefilter('ignore')


# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

num_intervals = snakemake.params.coalrate_epochs
num_mcmc = snakemake.params.mcmc_samples
num_burnin = snakemake.params.mcmc_burnin


# cross coalescence between strata
if snakemake.params.stratify is not None:
    for what in ('rates', 'pdf'):
        inp = pickle.load(open(snakemake.input.crossrate[0], "rb"))
        names, breaks = inp['names'], inp['breaks']
        ncol = names.size
        nrow = names.size
        fig, axs = plt.subplots(
            nrow, ncol, figsize=(3 * ncol, 2.5 * nrow), 
            sharex=True, sharey=True, squeeze=False,
            constrained_layout=True,
        )
        reps = 0
        mean = np.zeros_like(breaks)
        for i, f in enumerate(snakemake.input.crossrate[1:]):
            if i >= num_burnin:
                inp = pickle.load(open(f, "rb"))
                val = inp[what]
                mean += val
                reps += 1
                for j, p in enumerate(names):
                    for k, q in enumerate(names):
                        if j <= k:
                            axs[j, k].step(
                                breaks[j, k], 
                                val[j, k], 
                                color="gray",
                                alpha=0.1,
                            )
        mean /= reps
        for j, p in enumerate(names):
            for k, q in enumerate(names):
                if j <= k:
                    axs[j, k].step(
                        breaks[j, k], 
                        mean[j, k], 
                        color="black",
                        label=q, 
                        linewidth=1.5,
                    )
                    axs[j, k].set_xscale("log")
                    if what == "rates": axs[j, k].set_yscale("log")
                    if j == 0:
                        axs[j, k].set_xlabel(f"{q}")
                        axs[j, k].xaxis.set_label_position("top")
                    if k == j:
                        axs[j, k].set_ylabel(f"{p}")
                        axs[j, k].tick_params(labelbottom=True)
                        axs[j, k].tick_params(labelleft=True)
                else:
                    axs[j, k].set_visible(False)
        fig.supxlabel("Generations in the past")
        out_dir = os.path.dirname(snakemake.output.pair_coalescence_rates)
        if what == "rates":
            fig.supylabel("Pair coalescence rate")
            plt.savefig(os.path.join(out_dir, "cross-coalescence-rates.png"))
        elif what == "pdf":
            fig.supylabel("Proportion coalescencing pairs")
            plt.savefig(os.path.join(out_dir, "cross-coalescence-pdf.png"))
        plt.clf()


# pair coalescence rates with all samples
for what in ("pdf", "rates"):
    fig, axs = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    reps = 0
    mean = np.zeros(num_intervals)
    for i, f in enumerate(snakemake.input.coalrate):
        inp = pickle.load(open(f, "rb"))
        val, breaks = np.squeeze(inp[what]), np.squeeze(inp["breaks"])
        if i >= num_burnin:
            mean += val
            reps += 1
            axs.step(breaks, val, color="gray", alpha=0.1)
    mean /= reps
    axs.step(breaks, mean, color="black", linewidth=1.5)
    axs.set_xlabel("Generations in past")
    axs.set_xscale("log")
    if what == "rates": 
        axs.set_ylabel("Pair coalescence rate")
        axs.set_yscale("log")
        plt.savefig(snakemake.output.pair_coalescence_rates)
    elif what == "pdf":
        axs.set_ylabel("Proportion coalescing pairs")
        plt.savefig(snakemake.output.pair_coalescence_pdf)
    plt.clf()
