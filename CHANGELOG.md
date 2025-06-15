# 14-Jun-25 [0.0.1]

- Add provenance, populations to tree sequences
- Stratified statistics calculation for diagnostic plots
- `singer-binary` field in config has been removed; bundle a more recent SINGER binary
  (still version `0.1.8-beta` but with ratemap support)
- Uses ``tszip`` to compress tree sequences, use `tszip.load` to decompress
- Accept a list of filtered sites `<chrom>.filter.txt` and use this to adjust the mutation rate in addition
  to inaccessible intervals
