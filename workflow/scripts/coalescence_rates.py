"""
Calculate pair coalescence rates within time intervals from an ARG.
For statistical background, see https://github.com/tskit-dev/tskit/pull/2119

Part of https://github.com/nspope/singer-snakemake.
"""

import sys
import pickle
import msprime
import numpy as np
import tskit
from collections import defaultdict
from datetime import datetime

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


def weighted_pair_coalescence_rates(
    ts, sample_sets, indexes, windows, window_weights, 
    num_time_bins=25, log_time_bounds=[1, 7],
):
    """
    Calculate marginal pair coalescence rates, weighted by the proportion of
    missing data in each window and summed over windows
    """
    max_time = ts.nodes_time.max()
    time_windows = np.logspace(*log_time_bounds, num_time_bins + 1)
    pair_coalescence_counts = ts.pair_coalescence_counts(
        sample_sets=sample_sets,
        indexes=indexes,
        windows=windows,
        time_windows=time_windows,
        pair_normalise=True,
        span_normalise=True,
    )
    #assert np.all(pair_coalescence_counts >= 0.0)
    window_weights = window_weights * np.diff(windows)
    window_weights /= np.sum(window_weights)
    pair_coalescence_counts *= window_weights[:, np.newaxis, np.newaxis]
    pair_coalescence_counts = np.sum(pair_coalescence_counts, axis=0)
    # TODO: check
    pair_coalescence_surv = np.hstack([
        np.ones((pair_coalescence_counts.shape[0], 1)),
        1 - np.cumsum(pair_coalescence_counts, axis=1),
    ])
    pair_coalescence_rates = (
        np.log(pair_coalescence_surv[:, :-1]) - 
        np.log(pair_coalescence_surv[:, 1:])
    ) / np.diff(time_windows)[np.newaxis, :]
    pair_coalescence_rates[~np.isfinite(pair_coalescence_rates)] = np.nan
    # /TODO
    epochs = np.tile(time_windows[:-1], (pair_coalescence_counts.shape[0], 1))
    return pair_coalescence_rates.T, pair_coalescence_counts.T, epochs.T



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
weights = ratemap.rate # proportion missing times mutation rate
rates, pdf, breaks = weighted_pair_coalescence_rates(
    ts, sample_sets, indexes, windows, 
    weights, num_time_bins=num_intervals,
)
output = {
    "rates" : np.squeeze(rates), 
    "pdf" : np.squeeze(pdf),
    "breaks" : np.squeeze(breaks),
}
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
    cross_pdf = np.full((names.size, names.size, num_intervals), np.nan)
    cross_breaks = np.full((names.size, names.size, num_intervals), np.nan)
    for i, _ in enumerate(names):
        indexes = [(i, j) for j in range(names.size)]
        rates, pdf, breaks = \
            weighted_pair_coalescence_rates(
                ts, sample_sets, indexes, windows, 
                weights, num_time_bins=num_intervals,
            )
        cross_rates[i] = rates.T
        cross_pdf[i] = pdf.T
        cross_breaks[i] = breaks.T
    output = {
        "rates" : cross_rates, 
        "pdf" : cross_pdf,
        "breaks" : cross_breaks, 
        "names" : names,
    }
pickle.dump(output, open(snakemake.output.crossrate, "wb"))
