import numpy as np
import tskit
import msprime

from scipy.special import comb
from typing import List


def simulate_sequence_mask(ts: tskit.TreeSequence, density: float, length: float, seed: int = None):
    """
    Simulate a sequence mask by drawing interval locations from a Poisson process
    and interval lengths from a geometric distribution. Any missing flanks
    in the tree sequence will be included.
    """
    rng = np.random.default_rng(seed)
    bitmask = np.full(int(ts.sequence_length), False)
    left, right = int(ts.edges_left.min()), int(ts.edges_right.max())
    num_intervals = int(density * (right - left))
    interval_start = rng.integers(left, right, size=num_intervals)
    interval_length = rng.geometric(1.0 / length, size=num_intervals)
    for s, l in zip(interval_start, interval_length):
        bitmask[s:s+l] = True
    bitmask[:left] = True
    bitmask[right:] = True
    return bitmask


def simulate_variant_mask(ts: tskit.TreeSequence, proportion: float, seed: int = None):
    """
    Simulate a variant mask by picking sites uniformly at random.
    """
    rng = np.random.default_rng(seed)
    bitmask = rng.binomial(n=1, p=proportion, size=ts.num_sites).astype(bool)
    return bitmask


def ratemap_to_hapmap(ratemap: msprime.RateMap, contig_name: str, missing_as_zero: bool = False) -> str:
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


def assert_valid_hapmap(ratemap: msprime.RateMap, hapmap_path: str, ignore_missing: bool = False):
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
    np.testing.assert_allclose(bitmask, bitmask_check)


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
    sample_sets: List[List[int]], 
) -> (np.ndarray, np.ndarray):
    """
    Return mutation frequency in each sample set and the id of the
    edge carrying the mutation.
    """
    mutation_edge = np.full(ts.num_mutations, tskit.NULL)
    mutation_freq = np.full((ts.num_mutations, len(sample_sets)), tskit.NULL)
    for i, s in enumerate(sample_sets):
        for t in ts.trees(tracked_samples=s):
            for m in t.mutations():
                mutation_freq[m.id, i] = t.num_tracked_samples(m.node)
                mutation_edge[m.id] = m.edge
    return mutation_edge, mutation_freq


def time_windowed_mutation_afs(
    ts: tskit.TreeSequence, 
    sample_sets: List[List[int]], 
    time_breaks: np.ndarray,
    unknown_mutation_age: bool = True,
    span_normalise: bool = True,
) -> np.ndarray:
    """
    Calculate observed allele frequency spectrum in time windows. If the age of
    mutations is unknown, then integrate over mutation positions on a branch,
    such that the contribution of a mutation on a given edge to a given time
    window is (edge_overlap_with_window / length_of_edge).

    Output shape is (sfs_dimensions, num_time_windows).
    """
    assert time_breaks[0] == 0.0 and time_breaks[-1] == np.inf
    assert np.all(np.diff(time_breaks) > 0)

    # calculate mutation edge and frequency in populations
    edge, frequency = mutation_edge_and_frequency(ts, sample_sets)

    # remove mutations mapped above root
    keep = edge != tskit.NULL
    edge, frequency = edge[keep], frequency[keep]

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

    if span_normalise:
        afs /= ts.sequence_length

    return afs
