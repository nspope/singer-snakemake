"""
Test utilities related to allele polarisation and masking.
"""

import numpy as np
import msprime
import pytest
from workflow.validation.utils import repolarise_tree_sequence
from workflow.validation.utils import simulate_mispolarisation
from workflow.scripts.utils import absorb_mutations_above_root


def test_repolarise_tree_sequence():
    ts = msprime.sim_ancestry(
        samples=4, 
        sequence_length=100, 
        population_size=1, 
        recombination_rate=0, 
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=0.1, random_seed=2)
    biallelic = np.bincount(ts.mutations_site, minlength=ts.num_sites) == 1
    assert np.any(~biallelic)
    mispolarise = np.full_like(biallelic, False)
    mispolarise[~biallelic] = True
    mispolarise[:ts.num_sites // 2] = True
    geno_polar = repolarise_tree_sequence(ts, mispolarise).genotype_matrix()
    geno = ts.genotype_matrix()
    # biallelic sites that are flagged are flipped
    flipped = np.logical_and(mispolarise, biallelic)
    assert np.any(flipped)
    assert np.all(np.logical_xor(geno_polar[flipped], geno[flipped]))
    # all other sites are unchanged
    assert np.all(geno_polar[~flipped] == geno[~flipped])


def test_absorb_mutations_above_root():
    ts = msprime.sim_ancestry(
        samples=4, 
        sequence_length=100, 
        population_size=1, 
        recombination_rate=0, 
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=0.1, random_seed=2)
    mispolarise = np.full(ts.num_sites, False)
    mispolarise[:ts.num_sites // 2] = True
    ts_polarise = repolarise_tree_sequence(ts, mispolarise)
    ts_reverted = absorb_mutations_above_root(ts_polarise)
    geno_original = ts.genotype_matrix()
    geno_polarise = ts_polarise.genotype_matrix()
    geno_reverted = ts_reverted.genotype_matrix()
    assert np.any(geno_polarise != geno_original)
    assert np.all(geno_reverted == geno_original)


def test_major_allele_repolarisation():
    ts = msprime.sim_ancestry(
        samples=10, 
        sequence_length=1e5,
        population_size=1e4, 
        recombination_rate=1e-8, 
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=2)
    biallelic = np.bincount(ts.mutations_site, minlength=ts.num_sites) == 1
    ts = ts.delete_sites(~biallelic)
    ts = repolarise_tree_sequence(ts, simulate_mispolarisation(ts, "maf"))
    geno = ts.genotype_matrix()
    assert np.all(geno.sum(axis=-1) / ts.num_samples <= 0.5)


# TODO: test reference polarisation, frequency polarisation
# (or remove these options as not useful)


# --- masking WIP

# algorithm sketch:
# at each gap start, evaluate if max(edges_right) == gap
# so, get vector of edges_right length that is gap index.
# iterate over edges, every time a new edge is added update max(edges_right)
# if breakpoint corresponds to gap, and max(edges_right) == gap_start,
# then this is a gap to mask.
#
# we have a sequence of edges,
# |------|----|----|--------|
# |----|------|-------|-----|
# |-------------------------|
#          <-------------> gap
# --> remapped, will look like
# |------|-x--|
# |----|---x--|
#          <-------------> gap


def find_genealogical_gaps(ts, interval_breakpoints, interval_is_gap):
    """
    Find intervals in `accessible_ratemap` that have zero rate and across which
    no edges persist. If this condition is true, then the maximum of
    `ts.edges_right` for all edges to the left of the start of the gap will
    fall within the gap.
    """
    #interval_breakpoints = accessible_ratemap.position
    #prop_accessible = accessible_ratemap.rate
    #interval_is_gap = np.logical_or(prop_accessible == 0.0, np.isnan(prop_accessible))
    # TODO pass in the above
    num_intervals = interval_is_gap.size
    left, right = interval_breakpoints[:-1], interval_breakpoints[1:]
    sort_order = np.argsort(ts.edges_left)  # TODO: needed?
    left_index = np.digitize(ts.edges_left[sort_order], interval_breakpoints) - 1
    right_index = np.digitize(ts.edges_right[sort_order], interval_breakpoints) - 1
    assert left_index.min() > -1 and left_index.max() < num_intervals
    assert right_index.min() > -1 and right_index.max() < num_intervals
    right_index_max = np.maximum.accumulate(right_index)
    nonoverlapping = np.append(True, left_index[1:] >= right_index_max[:-1])
    complete_turnover = np.bincount(left_index[nonoverlapping], minlength=num_intervals).astype(bool)
    genealogical_gaps = np.logical_and(complete_turnover, interval_is_gap)
    intervals = np.stack([left[genealogical_gaps], right[genealogical_gaps]], axis=-1)
    return intervals


def check_interval_edge_overlap(ts, interval_breakpoints, interval_is_gap):
    """
    Find those `intervals` that, when subtracted from an edge,
    would split that edge in two.
    """
    sort_order = np.argsort(ts.edges_left) 
    left_index = np.digitize(ts.edges_left[sort_order], interval_breakpoints) - 1
    right_index = np.digitize(ts.edges_right[sort_order], interval_breakpoints, right=True) - 1
    for i, (a, b) in enumerate(zip(left_index, right_index)):
        assert a >= 0 and b < interval_is_gap.size
        assert a <= b
        print(ts.edges_left[sort_order][i], ts.edges_right[sort_order][i], left_index[i], right_index[i])
        #if a != b:
        #    is_gap = interval_is_gap[a:b + 1]
        #    if np.sum(is_gap[1:] != is_gap[:-1]) < 2:
        #        return False
    return True


def test_find_genealogical_gaps():
    ts = msprime.sim_ancestry(10, population_size=1e4, recombination_rate=1e-8, sequence_length=1e5)
    breaks = np.array([0, 50000, 60000, ts.sequence_length])
    is_gap = np.array([False, True, False])
    print(check_interval_edge_overlap(ts, breaks, is_gap))


# readme sketch:

# Accounting 
# One of the very useful things about ARGs is that we can compute expectations of summary statistics conditional on the ARG -- that is, without noise from the mutational process (these are the `branch` mode statistics in tskit, in constrast to `site` mode). Doing this correctly can be a bit tricky, however. There are three ways (that I know of) to account for masked sequence when calculating these expectations.
# - The first is to use `TreeSequence.delete_intervals`. However, edges that span a masked interval will be split. Hence if there are many small masked intervals (that are contained within trees) then using `delete_intervals` will lose much of the benefit of succinct tree sequences (compact size, fast calculation, edge spans that are informative about recombination and identity by descent). And it can be quite slow. So, we do not do this by default (but it can be toggled by adding XXXX in the config).
# - The second is to simulate mutations using a mutation rate map that has zero rate over masked intervals; then using `site` mode statistics. This is what we do to calculate diagnostics (so as to add uncertainty from the mutational process). This fine scale mutation rate map is stored as a pickled `RateMap` (XXXX). To calculate expectations, one would average over many simulations, but this is inefficient and will generally be noisy at a fine spatial scale.
# - The third is to map the tree sequence coordinate system to a new coordinate system that only counts accessible bases, and use `branch` mode in this new coordinate system. This is the best option, just mildly convoluted to implement. We do this to calculate coalescence rates and the genetic relatedness matrix (neither of which have a `site` mode).
#
# # Masking essential gaps
# The previous section has an argument against applying a fine scale mask directly with `TreeSequence.delete_intervals`. However, there is one type of masked interval that is beneficial to remove, which are gaps over which no edges persist (For each gap, find all edges that start to the left of the gap; if these edges end before or within the gap, then the gap can be removed without impacting the succinctness of the tree seqeunce). So, these gaps *are* deleted by default (this can be disabled by adding XXXX to the config).
#
# # Polarisation
# Another sticking point with actual data is polarisation. Typically variants are polarised relative to one or more outgroups. SINGER has an option to specify the probability that the ancestral allele is correct; so can be used for unpolarised data. In practice, though, the input polarisation seems to matter quite a lot. For example, the REF column in the unpolarised VCF will typically be the allele carried by the individual used for the reference assembly. Imagine the individual used to generate the reference genome is in the sample. Because it is the reference, all its genotypes in the (unpolarised) VCF will be 0. This is extremely unlikely under any realistic mutation model, and as a result SINGER will have a hard time escaping this "pathology" via MCMC. 
#
# This suggests a diagnostic for detecting polarisation error. In the absence of polarisation error or ascertainment bias, and if the mutational clock is constant, then all samples will have the same mutational load (count of derived mutations) in expectation. This is because the distance from each sample to the root is the same, in every tree. Hence, we can calculate mutational load per sample at each MCMC iteration (using whatever polarisation SINGER has settled on in that MCMC iteration); these should be more or less equal. The figure below shows a traceplot for the "reference genome" scenario described above.
#
# In practice, it seems that randomly choosing the ancestral allele for every site works well as a general purpose strategy for initializing the polarisations, so we do this by default if `polarised: False` (but this random initialization can be toggled off by adding XXXX to the config). The figure below shows a mutational load traceplot using the same data as before, but randomly initializing the polarizations.


def test_collapse_gaps():
    pass
    

