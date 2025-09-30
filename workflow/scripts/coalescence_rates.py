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

from utils import collapse_masked_intervals

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

min_pairs = 0.05
log_time_bounds, num_intervals = snakemake.params.time_grid[:2], snakemake.params.time_grid[2]  # TODO clean up
inaccessible = pickle.load(open(snakemake.input.inaccessible, "rb"))
ts = tszip.decompress(snakemake.input.trees)

time_windows = np.logspace(*log_time_bounds, num_intervals + 1)
time_windows = np.append(np.append(0, time_windows), np.inf)

# correct for masked sequence by adjusting edge spans
accessible = msprime.RateMap(position=inaccessible.position, rate=1 - inaccessible.rate)
ts = collapse_masked_intervals(ts, accessible)

# global pair coalescence rates
pdf = ts.pair_coalescence_counts(time_windows=time_windows, pair_normalise=True)
rates = ts.pair_coalescence_rates(time_windows=time_windows)
survival = np.append(1, 1 - np.cumsum(pdf))
rates[survival[:-1] <= min_pairs] = np.nan
output = {
    "rates" : rates[1:-1],
    "pdf" : pdf[1:-1],
    "breaks" : time_windows[1:-2],
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
        pdf = ts.pair_coalescence_counts(
            sample_sets=sample_sets,
            time_windows=time_windows,
            indexes=indexes,
            pair_normalise=True,
            span_normalise=True,
        )
        rates = ts.pair_coalescence_rates(
            sample_sets=sample_sets,
            time_windows=time_windows,
            indexes=indexes,
        )
        survival = np.concatenate([
            np.ones((names.size, 1)), 
            1 - np.cumsum(pdf, axis=-1)
        ], axis=-1)
        rates[survival[:, :-1] <= min_pairs] = np.nan
        cross_rates[i] = rates[:, 1:-1]
        cross_pdf[i] = pdf[:, 1:-1]
        cross_breaks[i] = np.tile(time_windows[1:-2], (names.size, 1))
    output = {
        "rates" : cross_rates, 
        "pdf" : cross_pdf,
        "breaks" : cross_breaks, 
        "names" : names,
    }
pickle.dump(output, open(snakemake.output.crossrate, "wb"))
