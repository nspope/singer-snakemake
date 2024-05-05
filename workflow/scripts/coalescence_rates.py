"""
Calculate pair coalescence rates within time intervals from an ARG.
For statistical background, see https://github.com/tskit-dev/tskit/pull/2119

Part of https://github.com/nspope/singer-snakemake.
"""

import sys
import pickle
import numba
import msprime
import numpy as np
import tskit
from collections import defaultdict
from datetime import datetime

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


@numba.njit("f8[:, :, :](i4[:, :], f8[:], f8[:, :], i4[:], i4[:], f8[:], f8[:], i4[:], i4[:], f8)")
def _pair_coalescence_counts(
    indexes, # (num_indexes, 2)
    windows, # (num_windows + 1)
    nodes_sample, # (num_nodes, len(sample_sets))
    edges_child,
    edges_parent,
    edges_left,
    edges_right,
    insert_index,
    remove_index,
    sequence_length,
):
    """
    Calculate pair coalescence rates in windows across the genome, within
    and between sample sets.

    This is a bastardization of `ts.pair_coalescence_counts` for numba:

    https://github.com/tskit-dev/tskit/issues/2904

    and will eventually be removed in favour of the native tskit method.
    """

    assert indexes.max() < nodes_sample.shape[1]
    assert edges_child.size == edges_parent.size == edges_left.size == edges_right.size
    assert edges_child.size == insert_index.size == remove_index.size
    assert windows.size > 1

    num_nodes = nodes_sample.shape[0]
    num_edges = edges_child.size
    num_windows = windows.size - 1
    num_indexes = indexes.shape[0]

    insert_position = edges_left[insert_index]
    remove_position = edges_right[remove_index]

    nodes_parent = np.full(num_nodes, tskit.NULL)
    coalescing_pairs = np.zeros((num_windows, num_nodes, num_indexes))
    sample_counts = nodes_sample.copy()

    position = 0.0
    w, a, b = 0, 0, 0
    while position < sequence_length:
        remainder = sequence_length - position

        while b < num_edges and remove_position[b] == position:  # edges out
            e = remove_index[b]
            p = edges_parent[e]
            c = edges_child[e]
            nodes_parent[c] = tskit.NULL
            inside = sample_counts[c]
            while p != tskit.NULL:
                outside = sample_counts[p] - sample_counts[c] - nodes_sample[p]
                for i, (j, k) in enumerate(indexes):
                    weight = inside[j] * outside[k] + inside[k] * outside[j]
                    coalescing_pairs[w, p, i] -= weight * remainder
                c, p = p, nodes_parent[p]
            p = edges_parent[e]
            while p != tskit.NULL:
                sample_counts[p] -= inside
                p = nodes_parent[p]
            b += 1

        while a < num_edges and insert_position[a] == position:  # edges in
            e = insert_index[a]
            p = edges_parent[e]
            c = edges_child[e]
            nodes_parent[c] = p
            inside = sample_counts[c]
            while p != tskit.NULL:
                sample_counts[p] += inside
                p = nodes_parent[p]
            p = edges_parent[e]
            while p != tskit.NULL:
                outside = sample_counts[p] - sample_counts[c] - nodes_sample[p]
                for i, (j, k) in enumerate(indexes):
                    weight = inside[j] * outside[k] + inside[k] * outside[j]
                    coalescing_pairs[w, p, i] += weight * remainder
                c, p = p, nodes_parent[p]
            a += 1

        position = sequence_length
        if b < num_edges:
            position = min(position, remove_position[b])
        if a < num_edges:
            position = min(position, insert_position[a])

        while w < num_windows and windows[w + 1] <= position:  # flush window
            remainder = sequence_length - windows[w + 1]
            for c, p in enumerate(nodes_parent):
                if p == tskit.NULL:
                    continue
                inside = sample_counts[c]
                outside = sample_counts[p] - sample_counts[c] - nodes_sample[p]
                for i, (j, k) in enumerate(indexes):
                    weight = inside[j] * outside[k] + inside[k] * outside[j]
                    coalescing_pairs[w, p, i] -= weight * remainder / 2
                    if w + 1 < num_windows:
                        coalescing_pairs[w + 1, p, i] += weight * remainder / 2
            w += 1

    for i, (j, k) in enumerate(indexes):
        if j == k:
            coalescing_pairs[:, :, i] /= 2

    for w, s in enumerate(windows[1:] - windows[:-1]):
        coalescing_pairs[w] /= s

    return coalescing_pairs


# TODO: invoke this inside numba above, 
def _rates_from_ecdf(weights, atoms, quantiles):
    assert weights.size == atoms.size
    assert np.all(weights > 0.0)
    assert np.all(atoms > 0.0)
    assert np.all(np.diff(atoms) >= 0.0)
    assert quantiles[0] == 0.0 and quantiles[-1] == 1.0
    assert quantiles.size > 2

    # find interior quantiles
    weights = np.append(0, weights)
    atoms = np.append(0, atoms)
    ecdf = np.cumsum(weights)
    indices = np.searchsorted(ecdf, ecdf[-1] * quantiles[1:-1], side='left')
    lower, upper = atoms[indices - 1], atoms[indices]
    ecdfl, ecdfu = ecdf[indices - 1] / ecdf[-1], ecdf[indices] / ecdf[-1]
    assert np.all(np.logical_and(quantiles[1:-1] > ecdfl,  quantiles[1:-1] < ecdfu))

    # interpolate ECDF
    assert np.all(ecdfu - ecdfl > 0)
    slope = (upper - lower) / (ecdfu - ecdfl)
    breaks = np.append(0, lower + slope * (quantiles[1:-1] - ecdfl))

    # calculate coalescence rates within intervals
    # this uses a Kaplan-Meier-type censored estimator: https://github.com/tskit-dev/tskit/pull/2119
    coalrate = np.full(quantiles.size - 1, np.nan)
    propcoal = np.diff(quantiles[:-1]) / (1 - quantiles[:-2])
    coalrate[:-1] = -np.log(1 - propcoal) / np.diff(breaks)
    last = indices[-1]
    coalrate[-1] = np.sum(weights[last:]) / np.dot(atoms[last:] - breaks[-1], weights[last:]) 

    return coalrate, breaks

def weighted_pair_coalescence_rates(ts, sample_sets, indexes, windows, window_weights, num_time_bins=50):
    """
    Calculate pair coalescence rates within equally-spaced quantiles over the
    chromosome, with a weighted contribution over genomic windows (ie weighted
    by missing data).  E.g. calculate the inverse of effective population size
    in N epochs wherein a proportion 1/N of pairs coalescece.
    """

    nodes_sample = np.zeros((ts.num_nodes, len(sample_sets)))
    for i, s in enumerate(sample_sets):
        nodes_sample[s, i] = 1

    indexes_array = np.zeros((len(indexes), 2), dtype=np.int32)
    for i, (j, k) in enumerate(indexes):
        indexes_array[i, :] = j, k

    # count number of coalescing pairs per node
    nodes_weight = _pair_coalescence_counts(
        indexes_array,
        windows,
        nodes_sample,
        ts.edges_child.copy(),
        ts.edges_parent.copy(),
        ts.edges_left.copy(),
        ts.edges_right.copy(),
        ts.indexes_edge_insertion_order.copy(),
        ts.indexes_edge_removal_order.copy(),
        ts.sequence_length,
    )
    
    nodes_order = np.argsort(ts.nodes_time)
    nodes_time = ts.nodes_time[nodes_order]
    nodes_weight = nodes_weight[:, nodes_order, :]

    # NB: global statistics must be weighted by the proportion of missing data
    nodes_weight *= window_weights[:, np.newaxis, np.newaxis]
    nodes_weight = nodes_weight.sum(axis=0)
    assert nodes_weight.shape[0] == ts.num_nodes
    assert nodes_weight.shape[1] == len(indexes)

    # convert to coalescence rates
    quantiles = np.linspace(0.0, 1.0, num_time_bins + 1)
    coalrate = np.full((num_time_bins, nodes_weight.shape[1]), np.nan)
    epochs = np.full((num_time_bins, nodes_weight.shape[1]), np.nan)
    for i in range(nodes_weight.shape[1]):
        inside_window = nodes_weight[:, i] > 0
        assert np.sum(inside_window) > 0, "No data in window"
        coalrate[:, i], epochs[:, i] = _rates_from_ecdf(
            nodes_weight[inside_window, i], 
            nodes_time[inside_window], 
            quantiles,
        )
            
    return coalrate, epochs


# NB: use this for windowed statistics in a separate script (TODO)
#def pair_coalescence_rates(ts, sample_sets, indexes, windows, num_time_bins=50):
#    """
#    Calculate pair coalescence rates within equally-spaced quantiles over a set
#    of genomic windows. E.g. calculate the inverse of effective population size
#    in N epochs wherein a proportion 1/N of pairs coalescece.
#    """
#
#    nodes_sample = np.zeros((ts.num_nodes, len(sample_sets)))
#    for i, s in enumerate(sample_sets):
#        nodes_sample[s, i] = 1
#
#    indexes_array = np.zeros((len(indexes), 2), dtype=np.int32)
#    for i, (j, k) in enumerate(indexes):
#        indexes_array[i, :] = j, k
#
#    # count number of coalescing pairs per node
#    nodes_weight = _pair_coalescence_counts(
#        indexes_array,
#        windows,
#        nodes_sample,
#        ts.edges_child.copy(),
#        ts.edges_parent.copy(),
#        ts.edges_left.copy(),
#        ts.edges_right.copy(),
#        ts.indexes_edge_insertion_order.copy(),
#        ts.indexes_edge_removal_order.copy(),
#        ts.sequence_length,
#    )
#    
#    nodes_order = np.argsort(ts.nodes_time)
#    nodes_time = ts.nodes_time[nodes_order]
#    nodes_weight = nodes_weight[:, nodes_order, :]
#
#    # convert to coalescence rates
#    quantiles = np.linspace(0.0, 1.0, num_time_bins + 1)
#    coalrate = np.full((nodes_weight.shape[0], num_time_bins, nodes_weight.shape[2]), np.nan)
#    epochs = np.full((nodes_weight.shape[0], num_time_bins, nodes_weight.shape[2]), np.nan)
#    for w in range(nodes_weight.shape[0]):
#        for i in range(nodes_weight.shape[2]):
#            inside_window = nodes_weight[w, :, i] > 0
#            if np.sum(inside_window) > 0:
#                coalrate[w, :, i], epochs[w, :, i] = _rates_from_ecdf(
#                    nodes_weight[w, inside_window, i], 
#                    nodes_time[inside_window], 
#                    quantiles,
#                )
#            
#    return coalrate, epochs


def simulation_test(seed, popsize, num_epochs):
    """
    For debugging, not run
    """
    return msprime.sim_ancestry(
        samples=100, 
        recombination_rate=1e-8, 
        sequence_length=20e6, 
        population_size=popsize, 
        random_seed=seed,
    )
    sample_sets = [list(ts.samples())]
    indexes = [(0, 0)]
    windows = np.linspace(0.0, ts.sequence_length, 2)
    est = np.squeeze(pair_coalescence_rates(ts, sample_sets, indexes, windows, num_epochs))
    print("Target coalesence rate:", 0.5 / popsize)
    print("Estimated coalesence rate in intervals:", est)
    print("Relative error:", (est - 0.5 / popsize) * popsize / 2)


# --- implm --- #

num_intervals = snakemake.params.coalrate_epochs
ts = tskit.load(snakemake.input.trees)
ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))

# global pair coalescence rates
sample_sets = [list(ts.samples())]
indexes = [(0, 0)]
windows = ratemap.position
weights = ratemap.rate / np.sum(ratemap.rate)
rates, breaks = weighted_pair_coalescence_rates(ts, sample_sets, indexes, windows, weights, num_intervals)
output = {"rates" : np.squeeze(rates), "breaks" : np.squeeze(breaks)}
pickle.dump(output, open(snakemake.output.coalrate, "wb"))

# stratified global pair coalescence rates (cross-coalescence)
output = {}
if snakemake.params.stratify is not None:
    sample_sets = defaultdict(list)
    for ind in ts.individuals():
        strata = ind.metadata[snakemake.params.stratify]
        sample_sets[strata].extend(ind.nodes)
    names = np.array(sorted(sample_sets.keys()))
    sample_sets = [sample_sets[x] for x in names]
    cross_rates = np.full((names.size, names.size, num_intervals), np.nan)
    cross_breaks = np.full((names.size, names.size, num_intervals), np.nan)
    for i, _ in enumerate(names):
        indexes = [(i, j) for j in range(names.size)]
        rates, breaks = \
            weighted_pair_coalescence_rates(ts, sample_sets, indexes, windows, weights, num_intervals)
        cross_rates[i] = rates.T
        cross_breaks[i] = breaks.T
    output = {
        "rates" : cross_rates, 
        "breaks" : cross_breaks, 
        "names" : names,
    }
pickle.dump(output, open(snakemake.output.crossrate, "wb"))
