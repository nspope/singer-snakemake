"""
Test utilities related to allele polarisation and masking.
"""

import numpy as np
import tskit
import msprime
import pytest
from workflow.scripts.validation.utils import repolarise_tree_sequence
from workflow.scripts.validation.utils import simulate_mispolarisation
from workflow.scripts.validation.utils import collapse_masked_intervals
from workflow.scripts.utils import absorb_mutations_above_root
from workflow.scripts.utils import find_genealogical_gaps 
from workflow.scripts.utils import compactify_run_length_encoding


def example_ts():
    """
	6┊         ┊         ┊     9   ┊    9    ┊
	 ┊         ┊         ┊   ┏━┻━┓ ┊  ┏━┻━┓  ┊
	5┊    8    ┊    8    ┊   8   ┃ ┊  8   ┃  ┊
	 ┊  ┏━┻━┓  ┊  ┏━┻━┓  ┊  ┏┻━┓ ┃ ┊ ┏┻┓  ┃  ┊
	4┊  ┃   ┃  ┊  ┃   ┃  ┊  ┃  ┃ ┃ ┊ ┃ ┃  7  ┊
	 ┊  ┃   ┃  ┊  ┃   ┃  ┊  ┃  ┃ ┃ ┊ ┃ ┃ ┏┻┓ ┊
	3┊  ┃   ┃  ┊  6   ┃  ┊  6  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
	 ┊  ┃   ┃  ┊ ┏┻┓  ┃  ┊ ┏┻┓ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
	2┊  ┃   5  ┊ ┃ ┃  5  ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
	 ┊  ┃  ┏┻┓ ┊ ┃ ┃ ┏┻┓ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
	1┊  4  ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
	 ┊ ┏┻┓ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊ ┃ ┃ ┃ ┃ ┊
	0┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 1 2 3 ┊ 0 2 1 3 ┊
	 0        10        20        30        40

    The fourth tree does not share edges with the first two trees;
    the third tree shares edges with the second and fourth trees.
    """
    tab = tskit.TableCollection(sequence_length=40)
    for i in range(4): tab.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
    for i in range(6): tab.nodes.add_row(time=i + 1)
    tab.edges.add_row(left=0, right=10, child=0, parent=4)
    tab.edges.add_row(left=0, right=10, child=1, parent=4)
    tab.edges.add_row(left=0, right=20, child=2, parent=5)
    tab.edges.add_row(left=0, right=20, child=3, parent=5)
    tab.edges.add_row(left=0, right=10, child=4, parent=8)
    tab.edges.add_row(left=0, right=20, child=5, parent=8)
    tab.edges.add_row(left=10, right=30, child=0, parent=6)
    tab.edges.add_row(left=10, right=30, child=1, parent=6)
    tab.edges.add_row(left=10, right=30, child=6, parent=8)
    tab.edges.add_row(left=20, right=40, child=2, parent=8)
    tab.edges.add_row(left=20, right=30, child=3, parent=9)
    tab.edges.add_row(left=20, right=40, child=8, parent=9)
    tab.edges.add_row(left=30, right=40, child=0, parent=8)
    tab.edges.add_row(left=30, right=40, child=1, parent=7)
    tab.edges.add_row(left=30, right=40, child=3, parent=7)
    tab.edges.add_row(left=30, right=40, child=7, parent=9)
    tab.sort()
    return tab.tree_sequence()


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


def test_find_genealogical_gaps_by_example():
    ts = example_ts()
    # example-specific "yes" intervals
    is_gap = np.array([False, True, False])
    genealogical_gaps = [
        # tree 4 is independent of 1,2 after deleting tree 3
        [20, 30],
        [15, 35],
        # trees 3,4 are independent of 1 after deleting tree 2
        [10, 20],
        [ 5, 25],
        # tree 4 is independent of 1 after deleting trees 2,3
        [ 5, 35],
    ]
    for gap in genealogical_gaps:
        breaks = np.array([0] + gap + [ts.sequence_length])
        np.testing.assert_allclose(find_genealogical_gaps(ts, breaks, is_gap), [gap])
    # example-specific "no" intervals
    is_gap = np.array([False, True, False])
    not_genealogical_gaps = [
        # deleting only a portion of trees 2,3 will not suffice
        [15, 25],
        # deleting only a portion of any tree will not suffice
        [ 5,  8],
        [15, 18],
        [25, 28],
        [35, 38],
    ]
    for gap in not_genealogical_gaps:
        breaks = np.array([0] + gap + [ts.sequence_length])
        np.testing.assert_allclose(find_genealogical_gaps(ts, breaks, is_gap), np.empty((0, 2)))
    # terminal intervals are always genealogical gaps
    np.testing.assert_allclose(
        find_genealogical_gaps(ts, np.array([0, 5, 40]), np.array([True, False])), 
        [[0, 5]],
    )
    np.testing.assert_allclose(
        find_genealogical_gaps(ts, np.array([0, 35, 40]), np.array([False, True])), 
        [[35, 40]],
    )
    # intervals that border terminal gaps are always genealogical gaps
    ts_gap = ts.delete_intervals([[0, 5], [35, 40]])
    np.testing.assert_allclose(
        find_genealogical_gaps(
            ts_gap, 
            np.array([0, 5, 8, 32, 35, 40]), 
            np.array([False, True, False, True, False])
        ), 
        [[5, 8], [32, 35]],
    )
    np.testing.assert_allclose(
        find_genealogical_gaps(
            ts_gap, 
            np.array([0, 8, 32, 40]), 
            np.array([True, False, True])
        ), 
        [[0, 8], [32, 40]],
    )
    # intervals that border internal gaps are always genealogical gaps
    ts_gap = ts.delete_intervals([[15, 25]])
    np.testing.assert_allclose(
        find_genealogical_gaps(
            ts_gap, 
            np.array([0, 12, 15, 25, 28, 40]), 
            np.array([False, True, False, True, False])
        ), 
        [[12, 15], [25, 28]],
    )
    ts_gap = ts.delete_intervals([[4, 6]])
    np.testing.assert_allclose(
        find_genealogical_gaps(
            ts_gap, 
            np.array([0, 2, 4, 6, 8, 40]), 
            np.array([False, True, False, True, False])
        ), 
        [[2, 4], [6, 8]],
    )
    # intervals that overlap external or internal gaps are always genealogical gaps
    ts_gap = ts.delete_intervals([[0, 5], [35, 40]])
    np.testing.assert_allclose(
        find_genealogical_gaps(
            ts_gap, 
            np.array([0, 3, 8, 32, 37, 40]), 
            np.array([False, True, False, True, False])
        ), 
        [[3, 8], [32, 37]],
    )
    ts_gap = ts.delete_intervals([[4, 8]])
    np.testing.assert_allclose(
        find_genealogical_gaps(
            ts_gap, 
            np.array([0, 6, 10, 40]), 
            np.array([False, True, False])
        ), 
        [[6, 10]],
    )


#@pytest.mark.skip
def test_find_genealogical_gaps_by_simulation():
    ts_gen = msprime.sim_ancestry(
        samples=10, 
        sequence_length=1e6, 
        recombination_rate=1e-8, 
        population_size=1e4, 
        num_replicates=100, 
        random_seed=1,
    )
    total_drop = 0
    total_keep = 0
    for ts in ts_gen:
        breaks = np.linspace(0, ts.sequence_length, 11)
        is_gap = np.array([False, True, False, False, True, True, False, True, True, False])
        gaps = find_genealogical_gaps(ts, breaks, is_gap)
        breaks, is_gap = compactify_run_length_encoding(breaks, is_gap)
        for i, (l, r) in enumerate(zip(breaks[:-1], breaks[1:])):
            if is_gap[i]:
                if [l, r] in gaps:
                    assert not np.any(np.logical_and(ts.edges_left < l, ts.edges_right > r))
                    total_drop += 1
                else:
                    assert np.any(np.logical_and(ts.edges_left < l, ts.edges_right > r))
                    total_keep += 1
    assert total_drop > 0
    assert total_keep > 0


def test_collapse_masked_intervals_by_example():
    ts = example_ts()
    accessible = msprime.RateMap(
        position=np.array([0., 5., 12., 18., 32., 40.]),
        rate=np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
    )
    ts_collapse = collapse_masked_intervals(ts, accessible)
    assert ts_collapse.num_trees == 3
    assert ts_collapse.num_edges == 15
    assert ts_collapse.num_nodes == ts.num_nodes
    assert ts_collapse.sequence_length == 19.0
    tree_spans = np.array([t.span for t in ts_collapse.trees()])
    tree_spans_ck = np.array([5., 6., 8.])
    np.testing.assert_allclose(tree_spans, tree_spans_ck)


#@pytest.mark.skip
def test_collapse_masked_intervals_by_simulation():
    num_intervals = 100
    rng = np.random.default_rng(1024)
    ts_gen = msprime.sim_ancestry(
        samples=10, 
        sequence_length=1e6, 
        recombination_rate=1e-8, 
        population_size=1e4, 
        num_replicates=100, 
        random_seed=1,
    )
    for ts in ts_gen:
        accessible = msprime.RateMap(
            position=np.append(np.append(0, np.sort(rng.uniform(size=num_intervals - 1))), 1) * ts.sequence_length,
            rate=rng.binomial(1, 0.5, size=num_intervals).astype(float),
        )
        intervals = np.stack([
            accessible.left[accessible.rate==0.0], 
            accessible.right[accessible.rate==0.0]
        ], axis=-1)
        ts_gap = ts.delete_intervals(intervals)
        ts_collapse = collapse_masked_intervals(ts, accessible)
        breaks = np.linspace(0, ts.sequence_length, 3)
        breaks_collapse = accessible.get_cumulative_mass(breaks)
        counts = ts_collapse.pair_coalescence_counts(windows=breaks_collapse)
        counts_ck = ts_gap.pair_coalescence_counts(windows=breaks)
        np.testing.assert_allclose(counts, counts_ck)


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
    

