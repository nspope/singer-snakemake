"""
Calculate observed and expected statistics from a tree sequence.  The latter is
done via simulation, so as to account for missing data without excessive
windowing.

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

from utils import mutational_load
from validation.utils import multiply_ratemaps 
from validation.utils import extract_accessible_ratemap


# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

windows = pickle.load(open(snakemake.input.windows, "rb"))
adjusted_mu = pickle.load(open(snakemake.input.mut_rate, "rb"))
inaccessible = pickle.load(open(snakemake.input.inaccessible, "rb"))
alleles = pickle.load(open(snakemake.input.alleles, "rb"))
trees = tszip.decompress(snakemake.input.trees) 
seed = 1 + int(snakemake.wildcards.rep) * 1000

accessible = msprime.RateMap(
    position=inaccessible.position,
    rate=1 - inaccessible.rate,
)
accessible = multiply_ratemaps(accessible, extract_accessible_ratemap(trees))
accessible_bp = np.diff(accessible.get_cumulative_mass(windows.position))

# remove sites that were omitted from dating
omitted = np.array([s.metadata["omitted"] for s in trees.sites()])
trees = trees.delete_sites(np.flatnonzero(omitted))

# number of flipped sites (only relevant for observed data)
repolarised = np.mean([s.metadata["flipped"] for s in trees.sites()])

# count multiple origins
multimapped = np.mean(np.bincount(trees.mutations_site, minlength=trees.num_sites))
# TODO: split by input frequency?

# simulate mutations given ARG topology and mask for posterior predictive checks
if snakemake.params.simulate_mutations:
    ts = msprime.sim_mutations(
        trees,
        rate=adjusted_mu, 
        random_seed=seed, 
        keep=False,
    )
else:
    ts = trees

diversity = ts.diversity(
    mode='site', 
    windows=windows.position, 
    span_normalise=False,
)
diversity[windows.rate == 1.0] /= accessible_bp[windows.rate == 1.0]
diversity[windows.rate == 0.0] = np.nan

segsites = ts.segregating_sites(
    mode='site', 
    windows=windows.position, 
    span_normalise=False,
)
segsites[windows.rate == 1.0] /= accessible_bp[windows.rate == 1.0]
segsites[windows.rate == 0.0] = np.nan

tajima_d = ts.Tajimas_D(
    mode='site', 
    windows=windows.position, 
)
tajima_d[windows.rate == 0.0] = np.nan

afs = ts.allele_frequency_spectrum(
    mode='site', 
    span_normalise=False,
    polarised=True,
) / accessible_bp[windows.rate == 1.0].sum()

# if there is no polarisation (or ascertainment) bias, then the number of mutations
# carried by each haplotype should be equal on average because the root-to-leaf length
# is the same across all leaves. 
load = mutational_load(ts)
load /= np.sum(accessible_bp)

stats = {
    "repolarised": repolarised,
    "multimapped": multimapped,
    "diversity": diversity, 
    "segsites": segsites,
    "tajima_d": tajima_d, 
    "afs": afs, 
    "load": load,
}
pickle.dump(stats, open(snakemake.output.stats, "wb"))


# TODO: clean this up
# stratified summary stats
strata_stats = {}
if snakemake.params.stratify is not None:
    sample_sets = {
        p.metadata["name"]: ts.samples(population=i) 
        for i, p in enumerate(ts.populations())
        if len(ts.samples(population=i))
    }
    strata = list(sorted(sample_sets.keys()))
    sample_sets = [sample_sets[x] for x in strata]
    strata_stats["strata"] = strata
    indexes = list(itertools.combinations_with_replacement(range(len(strata)), 2))

    strata_divergence = ts.divergence(
        # FIXME: this is a workaround for bug already fixed in tskit
        # https://github.com/tskit-dev/tskit/pull/3235, delete at some point
        sample_sets=sample_sets if len(sample_sets) > 1 else sample_sets * 2,
        indexes=indexes,
        mode='site',
        windows=windows.position,
        span_normalise=False,
    )
    strata_divergence[windows.rate == 1.0] /= \
        accessible_bp[windows.rate == 1.0, np.newaxis]
    strata_divergence[windows.rate == 0.0] = np.nan
    strata_stats["divergence"] = strata_divergence

    strata_afs = []
    for sample_set in sample_sets:
        afs = ts.allele_frequency_spectrum(
            sample_sets=[sample_set], 
            mode='site', 
            polarised=True,
            span_normalise=False,
        ) / accessible_bp[windows.rate == 1.0].sum()
        strata_afs.append(afs)
    strata_stats["afs"] = strata_afs

pickle.dump(strata_stats, open(snakemake.output.strata_stats, "wb"))
