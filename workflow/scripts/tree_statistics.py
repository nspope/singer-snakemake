"""
Calculate some branch statistics from a tree sequence.

Part of https://github.com/nspope/singer-snakemake.
"""

import msprime
import tskit
import tszip
import pickle
import numpy as np
import yaml
import itertools
from collections import defaultdict
from datetime import datetime


# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
ts = tszip.decompress(snakemake.input.trees)
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
    "diversity": diversity, 
    "tajima_d": tajima_d, 
    "folded_afs": folded_afs, 
    "unfolded_afs": unfolded_afs,
}
pickle.dump(stats, open(snakemake.output.stats, "wb"))

# TODO: clean this up
# stratified summary stats
strata_stats = {}
if snakemake.params.stratify is not None:
    sample_sets = defaultdict(list)
    for ind in ts.individuals():
        strata = ind.metadata[snakemake.params.stratify]
        sample_sets[strata].extend(ind.nodes)
    strata = [n for n in sample_sets.keys()]
    strata_stats["strata"] = strata

    strata_divergence = \
        ts.divergence(
            sample_sets=[s for s in sample_sets.values()], 
            indexes=list(itertools.combinations_with_replacement(range(len(strata)), 2)),
            mode='branch', windows=ratemap.position, span_normalise=True,
        ) * mutation_rate
    strata_divergence[ratemap.rate == 0.0] = np.nan
    strata_stats["divergence"] = strata_divergence

    strata_folded_afs = []
    for strata, sample_set in sample_sets.items():
        folded_afs = ts.allele_frequency_spectrum(
            sample_sets=[sample_set], mode='branch', 
            windows=ratemap.position, span_normalise=True,
        )
        folded_afs *= ratemap.rate[:, np.newaxis] / np.sum(ratemap.rate)
        folded_afs = np.sum(folded_afs, axis=0) * mutation_rate
        strata_folded_afs.append(folded_afs)
    strata_stats["folded_afs"] = strata_folded_afs

    strata_unfolded_afs = []
    for strata, sample_set in sample_sets.items():
        unfolded_afs = ts.allele_frequency_spectrum(
            sample_sets=[sample_set], mode='branch', polarised=True,
            windows=ratemap.position, span_normalise=True,
        )
        unfolded_afs *= ratemap.rate[:, np.newaxis] / np.sum(ratemap.rate)
        unfolded_afs = np.sum(unfolded_afs, axis=0) * mutation_rate
        strata_unfolded_afs.append(unfolded_afs)
    strata_stats["unfolded_afs"] = strata_unfolded_afs
pickle.dump(strata_stats, open(snakemake.output.strata_stats, "wb"))
