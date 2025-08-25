import numpy as np
import tszip

from utils import time_windowed_relatedness


time_windows = np.append(0, snakemake.params.time_grid)
for_individuals = snakemake.params.for_individuals
age_unknown = snakemake.params.unknown_mutation_age
span_normalise = snakemake.params.span_normalise

treefiles = iter(snakemake.input.trees)
ts = tszip.load(next(treefiles))

site_relatedness = time_windowed_relatedness(
    ts=ts, 
    time_breaks=time_windows, 
    unknown_mutation_age=age_unknown, 
    span_normalise=span_normalise,
    for_individuals=for_individuals,
)
for trees in treefiles:
    ts = tszip.load(trees)
    site_relatedness += \
        time_windowed_relatedness(
            ts=ts, 
            time_breaks=time_windows, 
            unknown_mutation_age=age_unknown, 
            span_normalise=span_normalise,
            for_individuals=for_individuals,
        )
site_relatedness /= len(snakemake.input.trees)

np.save(snakemake.output.site_relatedness, site_relatedness)
