# 29-Aug-25 [0.1.1]

- Use discretization scheme that delineates chunks between large "gaps" of masked sequence
- Use fine-scale recombination and mutation rate maps in SINGER
- Use POLEGON to date chunks accounting for masked sequence / missing data via branch spans
- Use branch midpoints as mutation times in tree sequence
- Absorb mutations above the root in inferred trees, switching ancestral state accordingly
- Add MCMC traceplot for proportion of such "flipped" mutations
- Add validation pipeline to simulate from a model then infer ARG and compare mutation ages
- Bugfix where population names were not stored correct in tree sequence population metadata
- Renamed some of the outputs for clarity

# 14-Jun-25 [0.1.0]

- Add provenance, populations to tree sequences
- Stratified statistics calculation for diagnostic plots
- `singer-binary` field in config has been removed; bundle a more recent SINGER binary
  (still version `0.1.8-beta` but with ratemap support)
- Uses ``tszip`` to compress tree sequences, use `tszip.load` to decompress
- Accept a list of filtered sites `<chrom>.filter.txt` and use this to adjust the mutation rate in addition
  to inaccessible intervals
