"""
Plot pair coalescence rates within time intervals from an ARG.

Part of https://github.com/nspope/singer-snakemake.
"""

import sys
import pickle
import numba
import msprime
import numpy as np
import matplotlib.pyplot as plt
import tskit

# --- lib --- #

def average_coalescence_rates(rates, breaks, quantiles):
    """
    Averaging rates is a little unintuitive: because the rates are a one-to-one
    function of the epoch widths (for the interior quantiles), we average the
    epoch widths and then recalculate rates. The rate in the terminal bin may
    be averaged directly.
    """
    mean_breaks = np.mean(breaks, axis=1)
    mean_rates = -np.log(1 - np.diff(quantiles[:-1]) / (1 - quantiles[:-2])) / np.diff(mean_breaks)
    mean_rates = np.append(mean_rates, np.mean(rates[-1]))
    return mean_rates, mean_breaks


# --- implm --- #

num_intervals = snakemake.params.coalrate_epochs
num_mcmc = snakemake.params.mcmc_samples
ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
windows = ratemap.position
quantiles = np.linspace(0, 1, num_intervals + 1)

# global pair coalescence rates
rates = np.full((num_intervals, num_mcmc), np.nan)
breaks = np.full((num_intervals, num_mcmc), np.nan)
for i, f in enumerate(snakemake.input.coalrate):
    inp = pickle.load(open(f, "rb"))
    rates[:, i] = np.squeeze(inp['rates'])
    breaks[:, i] = np.squeeze(inp['breaks'])
    plt.step(breaks[:, i], rates[:, i], color='gray', alpha=0.1)
mean_rates, mean_breaks = average_coalescence_rates(rates, breaks, quantiles)
plt.step(mean_breaks, mean_rates, color='black', linewidth=2)
plt.xlabel("Generations in past")
plt.ylabel("Pair coalescence rate")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig(snakemake.output.pair_coalescence_rates)
plt.clf()
