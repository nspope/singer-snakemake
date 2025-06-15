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

inaccessible = pickle.load(open(snakemake.input.inaccessible, "rb"))
filtered = pickle.load(open(snakemake.input.filtered, "rb"))
mutation_rate = snakemake.params.mutation_rate
chunk_size = np.diff(inaccessible.position)
accessible_size = chunk_size * (1 - inaccessible.rate)
ts = tszip.decompress(snakemake.input.trees)

diversity = \
    ts.diversity(mode='branch', windows=inaccessible.position, span_normalise=True) * \
    (1 - filtered.rate) * mutation_rate
diversity[inaccessible.rate == 1.0] = np.nan

# NB: these are per-chunk; correctly calculating at the sequence level
# would require calculating numerator and denominator separately, correcting,
# and summing across chunks
tajima_d = ts.Tajimas_D(mode='branch', windows=inaccessible.position)  # correction cancels within chunk
tajima_d[inaccessible.rate == 1.0] = np.nan

# NB: "global" statistics have to be weighted by accessible sequence length per chunk
correction = (1 - filtered.rate) * accessible_size / np.sum(accessible_size)

folded_afs = \
    ts.allele_frequency_spectrum(mode='branch', windows=inaccessible.position, span_normalise=True)
folded_afs = np.sum(folded_afs * correction[:, np.newaxis] * mutation_rate, axis=0)

unfolded_afs = \
    ts.allele_frequency_spectrum(mode='branch', windows=inaccessible.position, span_normalise=True, polarised=True) 
unfolded_afs = np.sum(unfolded_afs * correction[:, np.newaxis] * mutation_rate, axis=0)

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
            mode='branch', windows=inaccessible.position, span_normalise=True,
        ) * (1 - filtered.rate)[:, np.newaxis] * mutation_rate
    strata_divergence[inaccessible.rate == 0.0] = np.nan
    strata_stats["divergence"] = strata_divergence

    strata_folded_afs = []
    for strata, sample_set in sample_sets.items():
        folded_afs = ts.allele_frequency_spectrum(
            sample_sets=[sample_set], mode='branch', 
            windows=inaccessible.position, span_normalise=True,
        )
        folded_afs = np.sum(folded_afs * correction[:, np.newaxis] * mutation_rate, axis=0)
        strata_folded_afs.append(folded_afs)
    strata_stats["folded_afs"] = strata_folded_afs

    strata_unfolded_afs = []
    for strata, sample_set in sample_sets.items():
        unfolded_afs = ts.allele_frequency_spectrum(
            sample_sets=[sample_set], mode='branch', polarised=True,
            windows=inaccessible.position, span_normalise=True,
        )
        unfolded_afs = np.sum(unfolded_afs * correction[:, np.newaxis] * mutation_rate, axis=0)
        strata_unfolded_afs.append(unfolded_afs)
    strata_stats["unfolded_afs"] = strata_unfolded_afs
pickle.dump(strata_stats, open(snakemake.output.strata_stats, "wb"))
