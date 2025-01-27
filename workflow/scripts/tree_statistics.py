"""
Calculate some branch statistics from a tree sequence.

Part of https://github.com/nspope/singer-snakemake.
"""

import msprime
import tskit
import pickle
import numpy as np
import yaml
from datetime import datetime


# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
ts = tskit.load(snakemake.input.trees)
mutation_rate = snakemake.params.mutation_rate

diversity = \
    ts.diversity(mode='branch', windows=ratemap.position, span_normalise=True) * mutation_rate
diversity[ratemap.rate == 0.0] = np.nan

tajima_d = ts.Tajimas_D(mode='branch', windows=ratemap.position)
tajima_d[ratemap.rate == 0.0] = np.nan

# NB: "global" statistics have to be weighted by missing data per block
folded_afs = \
    ts.allele_frequency_spectrum(mode='branch', windows=ratemap.position, span_normalise=True)
folded_afs *= ratemap.rate[:, np.newaxis] / np.sum(ratemap.rate)
folded_afs = np.sum(folded_afs, axis=0) * mutation_rate

unfolded_afs = \
    ts.allele_frequency_spectrum(mode='branch', windows=ratemap.position, span_normalise=True, polarised=True) 
unfolded_afs *= ratemap.rate[:, np.newaxis] / np.sum(ratemap.rate)
unfolded_afs = np.sum(unfolded_afs, axis=0) * mutation_rate

stats = {
    "diversity" : diversity, 
    "tajima_d" : tajima_d, 
    "folded_afs" : folded_afs, 
    "unfolded_afs" : unfolded_afs,
}
pickle.dump(stats, open(snakemake.output.stats, "wb"))
