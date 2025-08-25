import numpy as np
import tszip

from string import ascii_lowercase, ascii_uppercase
from utils import hypergeometric_probabilities
from utils import time_windowed_afs


time_windows = np.append(0, snakemake.params.time_grid)
project_to = snakemake.params.project_to
age_unknown = snakemake.params.unknown_mutation_age
span_normalise = snakemake.params.span_normalise

treefiles = iter(snakemake.input.trees)
ts = tszip.load(next(treefiles))
sample_populations = np.unique([ts.nodes_population[i] for i in ts.samples()])
sample_sets = [ts.samples(population=p) for p in sample_populations]

site_afs = time_windowed_afs(
    ts=ts, 
    sample_sets=sample_sets, 
    time_breaks=time_windows, 
    unknown_mutation_age=age_unknown, 
    span_normalise=span_normalise,
)
for trees in treefiles:
    ts = tszip.load(trees)
    site_afs += \
        time_windowed_afs(
            ts=ts, 
            sample_sets=sample_sets, 
            time_breaks=time_windows, 
            unknown_mutation_age=age_unknown, 
            span_normalise=span_normalise,
        )
site_afs /= len(snakemake.input.trees)

if project_to is not None:  # down-project mutation frequencies
    projection = [hypergeometric_probabilities(len(s), project_to) for s in sample_sets]
    dim = len(projection)
    # einsum string has the form 'ijkt,iI,jJ,kK->IJKt'
    lhs = []
    lhs.append(''.join([x for x in ascii_lowercase[:dim]]) + 't')
    lhs.extend([x + y for x, y in zip(ascii_lowercase[:dim], ascii_uppercase[:dim])])
    lhs = ','.join(lhs)
    rhs = ''.join([x for x in ascii_uppercase[:dim]]) + 't'
    site_afs = np.einsum(f"{lhs}->{rhs}", site_afs, *projection)

np.save(snakemake.output.site_afs, site_afs)

