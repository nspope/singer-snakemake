"""
Calculate pair coalescence rates within time intervals from an ARG.
For statistical background, see https://github.com/tskit-dev/tskit/pull/2119

Part of https://github.com/nspope/singer-snakemake.
"""

import sys
import pickle
import msprime
import numpy as np
import tszip
from datetime import datetime

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


def weighted_pair_coalescence_rates(
    ts, sample_sets, indexes, *,
    windows=None, prop_accessible=None, 
    num_time_bins=25, log_time_bounds=[1, 7], cutoff=0.05,
):
    """
    Calculate marginal pair coalescence rates, weighted by the proportion of
    missing data in each window and summed over windows. Rates are not
    calculated where the proportion of uncoalesced pairs drops below `cutoff`.
    """
    if windows is None: 
        windows = np.array([0.0, ts.sequence_length])
    if prop_accessible is None: 
        prop_accessible = np.ones(windows.size - 1)
    max_time = ts.nodes_time.max()
    time_windows = np.logspace(*log_time_bounds, num_time_bins + 1)
    counts = ts.pair_coalescence_counts(
        sample_sets=sample_sets,
        indexes=indexes,
        windows=windows,
        time_windows=time_windows,
        pair_normalise=True,
        span_normalise=True,
    )
    #assert np.all(counts >= 0.0)
    weights = prop_accessible * np.diff(windows)
    weights /= np.sum(weights)
    counts *= weights[:, np.newaxis, np.newaxis]
    counts = np.sum(counts, axis=0)
    survival = np.hstack([
        np.ones((counts.shape[0], 1)), 
        1 - np.cumsum(counts, axis=1)
    ])
    intervals = np.diff(time_windows)
    rates = np.full_like(counts, np.nan)
    for s, r in zip(survival, rates):
        i = np.argmax(s <= cutoff)
        r[:i] = (np.log(s[:i]) - np.log(s[1:i+1])) / intervals[:i]
    epochs = np.tile(time_windows[:-1], (counts.shape[0], 1))
    return rates, counts, epochs


# --- implm --- #

num_intervals = snakemake.params.coalrate_epochs
inaccessible = pickle.load(open(snakemake.input.inaccessible, "rb"))
ts = tszip.decompress(snakemake.input.trees)

# correct for masked sequence by adjusting edge spans
accessible = msprime.RateMap(position=inaccessible.position, rate=1 - inaccessible.rate)
tab = ts.dump_tables()
tab.sites.clear()
tab.mutations.clear()
tab.edges.left = accessible.get_cumulative_mass(tab.edges.left)
tab.edges.right = accessible.get_cumulative_mass(tab.edges.right)
tab.edges.keep_rows(tab.edges.right > tab.edges.left)
ts = tab.tree_sequence().trim()
# TODO: functionalize and unit test the above
# TODO: the "weighted_xxxx" wrapper is unnecessary now

# global pair coalescence rates
sample_sets = [list(ts.samples())]
indexes = [(0, 0)]
rates, pdf, breaks = weighted_pair_coalescence_rates(
    ts, sample_sets, indexes, 
    num_time_bins=num_intervals,
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
    sample_sets = {
        p.metadata["name"]: ts.samples(population=i) 
        for i, p in enumerate(ts.populations())
        if len(ts.samples(population=i))
    }
    names = np.array(sorted(sample_sets.keys()))
    sample_sets = [sample_sets[x] for x in names]
    cross_rates = np.full((names.size, names.size, num_intervals), np.nan)
    cross_pdf = np.full((names.size, names.size, num_intervals), np.nan)
    cross_breaks = np.full((names.size, names.size, num_intervals), np.nan)
    for i, _ in enumerate(names):
        indexes = [(i, j) for j in range(names.size)]
        rates, pdf, breaks = \
            weighted_pair_coalescence_rates(
                ts, sample_sets, indexes, 
                num_time_bins=num_intervals,
            )
        cross_rates[i] = rates
        cross_pdf[i] = pdf
        cross_breaks[i] = breaks
    output = {
        "rates" : cross_rates, 
        "pdf" : cross_pdf,
        "breaks" : cross_breaks, 
        "names" : names,
    }
pickle.dump(output, open(snakemake.output.crossrate, "wb"))
