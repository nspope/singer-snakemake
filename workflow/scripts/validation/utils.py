"""
Utility functions for validation pipeline.

Part of https://github.com/nspope/singer-snakemake.
"""

import numpy as np
import tskit
import msprime

from scipy.special import comb
from typing import List


def simulate_sequence_mask(
    ts: tskit.TreeSequence, 
    density: float, 
    length: float, 
    seed: int = None,
) -> np.ndarray:
    """
    Simulate a sequence mask by drawing interval locations from a Poisson process
    and interval lengths from a geometric distribution. Any missing flanks
    in the tree sequence will be included.
    """
    rng = np.random.default_rng(seed)
    bitmask = np.full(int(ts.sequence_length), False)
    left, right = int(ts.edges_left.min()), int(ts.edges_right.max())
    bitmask[:left] = True
    bitmask[right:] = True
    if length > 0 and density > 0:
        num_intervals = int(density * (right - left))
        interval_start = rng.integers(left, right, size=num_intervals)
        interval_length = rng.geometric(1.0 / length, size=num_intervals)
        for s, l in zip(interval_start, interval_length):
            bitmask[s:s+l] = True
    return bitmask


def simulate_variant_mask(
    ts: tskit.TreeSequence, 
    proportion: float, 
    seed: int = None,
) -> np.ndarray:
    """
    Simulate a variant mask by picking sites uniformly at random.
    """
    rng = np.random.default_rng(seed)
    bitmask = rng.binomial(n=1, p=proportion, size=ts.num_sites).astype(bool)
    return bitmask


def ratemap_to_hapmap(
    ratemap: msprime.RateMap, 
    contig_name: str, 
    missing_as_zero: bool = False,
) -> str:
    """
    Write a recombination rate map into hapmap format.
    """
    physical_position = ratemap.position.astype(np.int64)
    scaled_rate = ratemap.rate * 1e8
    map_position = ratemap.get_cumulative_mass(physical_position) * 100
    hapmap = ["Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)"]
    if missing_as_zero: 
        scaled_rate[np.isnan(scaled_rate)] = 0.0
    if np.isnan(scaled_rate[-1]):  # handle trailing NaN
        scaled_rate[-1] = 0.0
    else:
        scaled_rate = np.append(scaled_rate, 0.0)
    for rate, pos, map in zip(scaled_rate, physical_position, map_position):
        if not np.isnan(rate): 
            hapmap.append(f"{contig_name}\t{pos}\t{rate:.10f}\t{map:.10f}")
    hapmap = "\n".join(hapmap) + "\n"
    return hapmap


def assert_valid_hapmap(
    ratemap: msprime.RateMap, 
    hapmap_path: str, 
    ignore_missing: bool = False,
):
    """
    Check that hapmap file matches rate map.
    """
    mask = np.full(ratemap.rate.size, True)
    if ignore_missing: mask[np.isnan(ratemap.rate)] = False
    hapmap_check = msprime.RateMap.read_hapmap(hapmap_path, rate_col=2, sequence_length=ratemap.sequence_length)
    np.testing.assert_allclose(hapmap_check.rate[mask], ratemap.rate[mask], atol=1e-12)
    np.testing.assert_allclose(hapmap_check.position, ratemap.position)
    hapmap_check = msprime.RateMap.read_hapmap(hapmap_path, map_col=3, sequence_length=ratemap.sequence_length)
    np.testing.assert_allclose(hapmap_check.rate[mask], ratemap.rate[mask], atol=1e-12)
    np.testing.assert_allclose(hapmap_check.position, ratemap.position)


def population_metadata_csv(ts: tskit.TreeSequence) -> (str, List[str]):
    """
    Extract population information per sample and return a csv with individual
    name/population, and a list of individual names with the format
    `<population_name>_<individual id>`.
    """
    samples = list(ts.samples())
    sample_populations = np.unique(ts.nodes_population[samples])
    population_names = [p.metadata.get("name", p.id) for p in ts.populations()]
    individual_names = []
    metadata_csv = ['"id","population"']
    for i in ts.individuals():
        pop = population_names[i.population]
        name = f"{pop}_{i.id}"
        individual_names.append(name)
        metadata_csv.append(f'"{name}","{pop}"')
    metadata_csv = "\n".join(metadata_csv) + "\n"
    return metadata_csv, individual_names


def bitmask_to_bed(bitmask: np.ndarray, contig_name: str) -> str:
    """
    Convert a bitmask to BED format.
    """
    changepoints = np.flatnonzero(bitmask[1:] != bitmask[:-1])
    changepoints = np.append(np.append(0, changepoints + 1), bitmask.size)
    bedmask = []
    for s, e in zip(changepoints[:-1], changepoints[1:]):
        if bitmask[s]: bedmask.append(f"{contig_name}\t{s}\t{e}")
    bedmask = "\n".join(bedmask) + "\n"
    return bedmask


def assert_valid_bedmask(bitmask: np.ndarray, bedmask_path: str):
    """
    Check that bed file matches bitmask.
    """
    bitmask_check = np.full_like(bitmask, False)
    bedmask = np.loadtxt(bedmask_path, usecols=[1, 2], dtype=int)
    for a, b in bedmask: bitmask_check[a:b] = True
    assert np.all(~np.logical_xor(bitmask, bitmask_check))


def bed_to_bitmask(bedmask_path: str, bitmask: np.ndarray):
    """
    Apply bed to bitmask (modified in place).
    """
    if bedmask_path is None: return
    bedmask = np.loadtxt(bedmask_path, usecols=[1, 2]) 
    for (a, b) in bedmask.astype(np.int64): bitmask[a:b] = True


def simulate_mispolarisation(
    ts: tskit.TreeSequence,
    proportion: [str|float],
    seed: int = None,
):
    """
    Choose which sites to mispolarise. If `proportion` is a float, select sites
    at random at this proportion.  If `maf`, polarise by major allele. If
    `ref`, choose a haplotype at random and repolarise any site where it
    carries the derived allele.
    """
    rng = np.random.default_rng(seed)
    if proportion == "maf":  # set major allele to ancestral
        variant_frequency = np.zeros(ts.num_sites)
        for t in ts.trees():
            for s in t.sites():
                if len(s.mutations) == 1:
                    m = next(iter(s.mutations))
                    variant_frequency[s.id] = t.num_samples(m.node)
        variant_frequency /= ts.num_samples
        mispolarise = variant_frequency > 0.5
    elif proportion == "ref":  # set alleles carried by a random haplotype to ancestral
        ref = rng.integers(0, ts.num_samples, size=1)
        mispolarise = np.full(ts.num_sites, False)
        for t in ts.trees(sample_lists=True):
            for s in t.sites():
                if len(s.mutations) == 1:
                    m = next(iter(s.mutations))
                    for i in t.samples(m.node):
                        if i == ref:
                            mispolarise[s.id] = True
    elif proportion == "frq":  # set alleles ancestral with prop equal to their frequency
        variant_frequency = np.zeros(ts.num_sites)
        for t in ts.trees():
            for s in t.sites():
                if len(s.mutations) == 1:
                    m = next(iter(s.mutations))
                    variant_frequency[s.id] = t.num_samples(m.node)
        variant_frequency /= ts.num_samples
        mispolarise = rng.binomial(1, variant_frequency) > 0
    else:  # set a random proportion of alleles to ancestral
        assert 0 <= proportion <= 1
        mispolarise = rng.binomial(1, proportion, size=ts.num_sites) > 0
    return mispolarise


def repolarise_tree_sequence(
    ts: tskit.TreeSequence, 
    repolarise: np.ndarray,
) -> tskit.TreeSequence:
    """
    Flip polarisation of site if `repolarise[site.id]`. Skip multiallelic and
    nonsegregating sites.
    """
    assert repolarise.size == ts.num_sites
    tab = ts.dump_tables()
    tab.mutations.clear()
    tab.sites.clear()
    tree = ts.first()
    for site in ts.sites():
        tree.seek(site.position)
        biallelic = len(site.mutations) == 1
        if repolarise[site.id] and biallelic:
            mutation = next(iter(site.mutations))
            for r in tree.roots:
                tab.mutations.add_row(site=site.id, node=r, derived_state=site.ancestral_state)
            tab.sites.add_row(position=site.position, ancestral_state=mutation.derived_state)
        else:
            tab.sites.add_row(position=site.position, ancestral_state=site.ancestral_state)
        for mutation in site.mutations:
            tab.mutations.add_row(site=mutation.site, node=mutation.node, derived_state=mutation.derived_state)
    tab.sort()
    tab.build_index()
    tab.compute_mutation_parents()
    assert tab.sites.num_rows == ts.num_sites
    assert np.all(tab.sites.position == ts.sites_position)
    return tab.tree_sequence()


def collapse_masked_intervals(
    ts: tskit.TreeSequence, 
    accessible: msprime.RateMap,
) -> tskit.TreeSequence:
    """
    Return a copy of the tree sequence with masked intervals (where `accessible.rate == 0.0`)
    collapsed, so that the coordinate system is in terms of unmasked sequence length.
    Zero length edges are removed, and any nodes that are then disconnected are removed as well.
    All sites and mutations that are within the collapsed intervals are removed.
    """
    assert np.all(np.logical_or(accessible.rate == 0.0, accessible.rate == 1.0))
    assert accessible.sequence_length == ts.sequence_length
    tab = ts.dump_tables()
    tab.sequence_length = accessible.get_cumulative_mass(ts.sequence_length)
    # map edges to new coordinate system and remove those with zero length
    tab.edges.left = accessible.get_cumulative_mass(tab.edges.left)
    tab.edges.right = accessible.get_cumulative_mass(tab.edges.right)
    tab.edges.keep_rows(tab.edges.right > tab.edges.left)
    # remove disconnected nodes
    is_connected = np.full(tab.nodes.num_rows, False)
    is_connected[tab.edges.parent] = True
    is_connected[tab.edges.child] = True
    node_map = tab.nodes.keep_rows(is_connected)
    tab.edges.parent = node_map[tab.edges.parent]
    tab.edges.child = node_map[tab.edges.child]
    # map sites to new coordinate system and remove those in masked intervals
    site_map = tab.sites.keep_rows(accessible.get_rate(tab.sites.position).astype(bool))
    tab.sites.position = accessible.get_cumulative_mass(tab.sites.position)
    # update mutation pointers and remove those without a node or site
    tab.mutations.node = node_map[tab.mutations.node]
    tab.mutations.site = site_map[tab.mutations.site]
    tab.mutations.keep_rows(
        np.logical_and(
            tab.mutations.site != tskit.NULL, 
            tab.mutations.node != tskit.NULL,
        )
    )
    tab.sort()
    tab.build_index()
    tab.compute_mutation_parents()
    return tab.tree_sequence()


def hypergeometric_probabilities(input_dim: int, output_dim: int) -> np.ndarray:
    """
    Matrix of hypergeometric sampling probabilities; the `i,j`th element is the
    probability of sampling `j` derived variants in a subsample of size
    `output_dim` ploids, given that there are originally `i` derived variants in a
    sample of size `input_dim` ploids.
    """
    assert input_dim >= output_dim
    prob = np.zeros((input_dim + 1, output_dim + 1))
    for i in range(input_dim + 1):
        for j in range(output_dim + 1):
            prob[i, j] = \
                comb(i, j, exact=True) * \
                comb(input_dim - i, output_dim - j, exact=True) / \
                comb(input_dim, output_dim, exact=True)
    return prob


def mutation_edge_and_frequency(
    ts: tskit.TreeSequence,
    sample_sets: List[List[int]] = None, 
) -> (np.ndarray, np.ndarray):
    """
    Return mutation frequency (number of carriers) in each sample set and the
    id of the edge carrying the mutation.
    """
    if sample_sets is None:
        sample_sets = [list(ts.samples())]
    mutation_edge = np.full(ts.num_mutations, tskit.NULL)
    mutation_freq = np.full((ts.num_mutations, len(sample_sets)), tskit.NULL)
    for i, s in enumerate(sample_sets):
        for t in ts.trees(tracked_samples=s):
            for m in t.mutations():
                mutation_freq[m.id, i] = t.num_tracked_samples(m.node)
                mutation_edge[m.id] = m.edge
    return mutation_edge, mutation_freq


def ancestral_state_and_frequency(
    ts: tskit.TreeSequence,
    sample_sets: List[List[int]] = None, 
) -> (np.ndarray, np.ndarray):
    """
    Return ancestral state of each site and its frequency (number of carriers)
    in each sample set.
    """
    if sample_sets is None:
        sample_sets = [list(ts.samples())]
    state = np.empty(ts.num_sites, dtype=object)
    freq = np.full((ts.num_sites, len(sample_sets)), tskit.NULL)
    for i, s in enumerate(sample_sets):
        for v in ts.variants(samples=s):
            uid, anc = v.site.id, v.site.ancestral_state
            state[uid] = anc
            freq[uid, i] = v.counts()[anc]
    return state, freq


def time_windowed_afs(
    ts: tskit.TreeSequence, 
    sample_sets: List[List[int]], 
    time_breaks: np.ndarray,
    unknown_mutation_age: bool = True,
) -> np.ndarray:
    """
    Calculate observed allele frequency spectrum in time windows. If the age of
    mutations is unknown, then integrate over mutation positions on a branch,
    such that the contribution of a mutation on a given edge to a given time
    window is `(edge_overlap_with_window / length_of_edge)`. Only sites carrying a
    single mutation are used.

    Output shape is `(sfs_dimensions, num_time_windows)`.
    """
    assert time_breaks[0] == 0.0 and time_breaks[-1] == np.inf
    assert np.all(np.diff(time_breaks) > 0)

    # calculate mutation edge and frequency in populations
    edge, frequency = mutation_edge_and_frequency(ts, sample_sets)

    # remove mutations mapped above root
    keep = edge != tskit.NULL
    edge, frequency = edge[keep], frequency[keep]
    assert np.all(frequency != tskit.NULL)

    # find time window per mutation
    node_below = ts.edges_child[edge]
    node_above = ts.edges_parent[edge]
    window_below = np.digitize(ts.nodes_time[node_below], time_breaks) - 1
    window_above = np.digitize(ts.nodes_time[node_above], time_breaks) - 1
    assert np.all(np.logical_and(window_below >= 0, window_below < time_breaks.size - 1))
    assert np.all(np.logical_and(window_above >= 0, window_above < time_breaks.size - 1))

    # calculate overlap with time window beneath node
    overlap_below = ts.nodes_time[node_below] - time_breaks[window_below] 
    overlap_above = ts.nodes_time[node_above] - time_breaks[window_above] 

    # calculate time windowed AFS
    dim = tuple([len(s) + 1 for s in sample_sets])
    afs = np.zeros((*dim, time_breaks.size - 1))
    if unknown_mutation_age:  # FIXME: this is not particuarly efficient
        density = 1 / (ts.nodes_time[node_above] - ts.nodes_time[node_below])
        length = np.diff(time_breaks)
        for i, f in enumerate(frequency):
            f = tuple(f)
            s, e = window_below[i], window_above[i]
            afs[*f][s:e] += length[s:e] * density[i]
            afs[*f][s] -= overlap_below[i] * density[i]
            afs[*f][e] += overlap_above[i] * density[i]
    else:
        time_index = np.digitize(ts.mutations_time[keep], time_breaks) - 1
        assert np.all(np.logical_and(time_index >= 0, time_index < time_breaks.size - 1))
        for t, f in zip(time_index, frequency): 
            afs[*f][t] += 1.0

    return afs


def time_windowed_relatedness(
    ts: tskit.TreeSequence,
    time_breaks: np.ndarray,
    unknown_mutation_age: bool = True,
    for_individuals: bool = True,
):
    """
    Calculate observed genetic relatedness (the average number of shared
    mutations between pairs of haplotypes) in time windows. If the age of
    mutations is unknown, then integrate over mutation positions on a branch,
    such that the contribution of a mutation on a given edge to a given time
    window is `(edge_overlap_with_window / length_of_edge)`. Only sites carrying a
    single mutation are used.

    Output shape is `(num_samples, num_samples, num_time_windows)` or
    `(num_individuals, num_individuals, num_time_windows)` if `for_individuals`.
    """
    assert time_breaks[0] == 0.0 and time_breaks[-1] == np.inf
    assert np.all(np.diff(time_breaks) > 0)

    # find mutation mapping to edge 
    edge, _ = mutation_edge_and_frequency(ts)
    keep = edge != tskit.NULL

    # find time windows of edge endpoints
    node_below = ts.edges_child[edge]
    node_above = ts.edges_parent[edge]
    window_below = np.digitize(ts.nodes_time[node_below], time_breaks) - 1
    window_above = np.digitize(ts.nodes_time[node_above], time_breaks) - 1
    assert np.all(np.logical_and(window_below >= 0, window_below < time_breaks.size - 1))
    assert np.all(np.logical_and(window_above >= 0, window_above < time_breaks.size - 1))

    # calculate overlap with time window beneath node
    overlap_below = ts.nodes_time[node_below] - time_breaks[window_below] 
    overlap_above = ts.nodes_time[node_above] - time_breaks[window_above] 
    density = 1 / (ts.nodes_time[node_above] - ts.nodes_time[node_below])

    # calculate time windowed relatedness
    relatedness = np.zeros((time_breaks.size - 1, ts.num_samples, ts.num_samples))
    if unknown_mutation_age:  
        correction = np.zeros_like(relatedness)
        for t in ts.trees(sample_lists=True):
            for m in t.mutations():
                i = m.id
                if keep[i]:
                    u, v = window_below[i], window_above[i]
                    s = list(t.samples(m.node))
                    pairs = np.ix_(s, s)
                    relatedness[v][pairs] += overlap_above[i] * density[i]
                    relatedness[u][pairs] -= overlap_below[i] * density[i]
                    correction[v][pairs] -= density[i]
                    correction[u][pairs] += density[i]
        relatedness[:-1] += correction[:-1].cumsum(axis=0) * \
            np.diff(time_breaks)[:-1, np.newaxis, np.newaxis]
    else:
        time_index = np.digitize(ts.mutations_time, time_breaks) - 1
        assert np.all(np.logical_and(time_index >= 0, time_index < time_breaks.size - 1))
        for t in ts.trees(sample_lists=True):
            for m in t.mutations():
                i = m.id
                if keep[i]:
                    u = time_index[i]
                    s = list(t.samples(m.node))
                    relatedness[u][np.ix_(s, s)] += 1.0

    if for_individuals:  # average over haplotypes for pairs of individuals
        individual_map = np.zeros((ts.num_individuals, ts.num_samples))
        for i in ts.individuals():
            individual_map[i.id, i.nodes] = 1 / i.nodes.size
        relatedness = individual_map @ relatedness @ individual_map.T

    return relatedness.transpose()

    
