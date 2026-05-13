# [0.1.2]

- Use ancestral states if provided as fasta; sites with ambiguous ancestral states are treated as unpolarised
- Include a custom version of SINGER binary that allows per-site polarisation
- Add `singer-skip-failures` flag that will treat chunks where MCMC fails as missing and still produce output
- Use per-sample masks if provdied; these are always hard-masked leaving trees with varying numbers of tips
- All outputs in one-based coords; previously ratemaps were zero-based while tree sequences were one-based
- Option to average coalescence cdf/rates over recombination map rather than bp, which can reduce variance
- Switch from hardcoded binary versions to checksums in provenance

TODO: 
  * document per-sample-masks paircoal-reweight and ancestral fasta in README
  * expose hard-masking

# 3-Mar-26 [0.1.1]

- Use discretization scheme that delineates chunks between large "gaps" of masked sequence
- Use POLEGON to date chunks accounting for masked sequence / missing data via branch spans
- Use branch midpoints as mutation times in tree sequence
- Absorb mutations above the root in inferred trees, switching ancestral state accordingly
- Add MCMC traceplot for proportion of such "flipped" mutations
- Add validation pipeline to simulate from a model then infer ARG and compare mutation ages
- Bugfix where population names were not stored correct in tree sequence population metadata
- Renamed some of the outputs for clarity
- Remove masked intervals where trees are uncorrelated on either flank
- Finer spatial resolution in some stats checks

# 14-Jun-25 [0.1.0]

- Add provenance, populations to tree sequences
- Stratified statistics calculation for diagnostic plots
- `singer-binary` field in config has been removed; bundle a more recent SINGER binary
  (still version `0.1.8-beta` but with ratemap support)
- Uses ``tszip`` to compress tree sequences, use `tszip.load` to decompress
- Accept a list of filtered sites `<chrom>.filter.txt` and use this to adjust the mutation rate in addition
  to inaccessible intervals
