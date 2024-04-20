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
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coalescence-rates", nargs="+", type=str, required=True)
    parser.add_argument("--pair-coalescence-rates-plot", type=str, required=True)
    args = parser.parse_args()

    for i, f in enumerate(args.coalescence_rates):
        inp = pickle.load(open(f, "rb"))
        rates = np.squeeze(inp['rates'])
        breaks = np.squeeze(inp['breaks'])
        plt.step(breaks, rates, color='black', alpha=0.2)
    plt.xlabel("Generations in past")
    plt.ylabel("Pair coalescence rate")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(args.pair_coalescence_rates_plot)

    sys.exit(0)
