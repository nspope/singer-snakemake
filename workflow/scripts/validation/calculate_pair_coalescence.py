import numpy as np
import msprime
import itertools
import tszip

from utils import collapse_masked_intervals


time_windows = np.append(0, snakemake.params.time_grid)
inaccessible = pickle.load(open(snakemake.input.inaccessible, "rb"))
accessible = msprime.RateMap(position=inaccessible.position, rate=1 - inaccessible.rate)
treefiles = iter(snakemake.input.trees)

ts = collapse_masked_intervals(tszip.load(next(treefiles)), accessible)
sample_populations = np.unique([ts.nodes_population[i] for i in ts.samples()])
sample_sets = [ts.samples(population=p) for p in sample_populations]
num_sample_sets = len(sample_sets)
dim = (num_sample_sets, num_sample_sets, time_windows.size - 1)
indexes = list(itertools.product(range(num_sample_sets), range(num_sample_sets)))

pair_density = ts.pair_coalescence_counts(
    sample_sets=sample_sets, 
    indexes=indexes,
    time_windows=time_windows, 
    pair_normalise=True,
).reshape(dim)
pair_rates = ts.pair_coalescence_rates(
    sample_sets=sample_sets, 
    indexes=indexes,
    time_windows=time_windows, 
).reshape(dim)
for trees in treefiles:
    ts = collapse_masked_intervals(tszip.load(trees), accessible)
    pair_density += ts.pair_coalescence_counts(
        sample_sets=sample_sets, 
        indexes=indexes,
        time_windows=time_windows, 
        pair_normalise=True,
    ).reshape(dim)
    pair_rates += ts.pair_coalescence_rates(
        sample_sets=sample_sets, 
        indexes=indexes,
        time_windows=time_windows, 
    ).reshape(dim)
pair_density /= len(snakemake.input.trees)
pair_rates /= len(snakemake.input.trees)

np.save(snakemake.output.pair_density, pair_density)
np.save(snakemake.output.pair_rates, pair_rates)
