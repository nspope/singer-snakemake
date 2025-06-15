This is a Snakemake workflow for running
[SINGER](https://github.com/popgenmethods/SINGER) (MCMC sampling of ancestral
recombination graphs) in parallel (e.g. across chunks of sequence). The genome
is discretized into chunks, and SINGER is run on each chunk with parameters
adjusted to account for missing sequence and recombination rate heterogeneity.
Chunks are merged into a single tree sequence per chromosome and MCMC
replicate. Some diagnostic plots are produced at the end, that compare summary
statistics to their expectations given the ARG topology. Pair coalescence rates
are calculated from the tree sequences and plotted. 

Please cite SINGER (https://doi.org/10.1101/2024.03.16.585351) if you use this
pipeline. Note that I'm not one of the authors of SINGER, and put this pipeline
together for my own research, so *caveat emptor*.

### Dependencies

Using `git` and `mamba` and `pip`:

```bash
git clone https://github.com/nspope/singer-snakemake my-singer-run && cd my-singer-run
mamba env create -f environment.yaml 
mamba activate singer-snakemake-env
snakemake --cores=20 --configfile=config/example_config.yaml
```

### Inputs

The input files for each chromosome are:

  - __chromosome_name.vcf.gz__ gzip'd VCF that can be used as SINGER input, either diploid and **phased** or haploid with an even number of samples
  - __chromosome_name.mask.bed__ (optional) bed file containing inaccessible intervals
  - __chromosome_name.hapmap__ (optional) recombination map in the format described in the documentation for `msprime.RateMap.read_hapmap` (see [here](https://tskit.dev/msprime/docs/stable/api.html#msprime.RateMap.read_hapmap))
  - __chromosome_name.meta.csv__ (optional) csv containing metadata for each sample in the VCF, that will be inserted into the output tree sequences. The first row should be the field names, with subsequent rows for every sample in the VCF.
  - __chromosome_name.filter.txt__ (optional) text file containing 1-based positions of filtered SNPs (one per line). These may be present in the VCF (in which case they are removed by the pipeline) or not (in which case they are still used to calculate an adjustment to mutation rate, see below).

see `example/*`.

### Config

A template for the configuration file is in `configs/example_config.yaml`:

```yaml
# --- example_config.yaml ---
input-dir: "example" # directory with input files per chromosome, that are "chrom.vcf" "chrom.hapmap" "chrom.mask.bed"
chunk-size: 1e6 # target size in base pairs for each singer run
max-missing: 0.975 # ignore chunks with more than this proportion of missing bases
mutation-rate: 1e-8 # per base per generation mutation rate
recombination-rate: 1e-8 # per base per generation recombination rate, ignored if hapmap is present
polarised: True # are variants polarised so that the reference state is ancestral
mcmc-samples: 100 # number of MCMC samples (each sample is a tree sequence)
mcmc-thin: 10 # thinning interval between MCMC samples
mcmc-burnin: 0.2 # proportion of initial samples discarded when computing plots of statistics
mcmc-resumes: 1000 # maximum number of times to try to resume MCMC on error at a given iteration
coalrate-epochs: 25 # number of time intervals within which to calculate statistics
stratify-by: "population" # stratify cross coalescence rates by this column in the metadata, or None
random-seed: 1 # random seed
```

### Missing data, mutation rate, and dating

For the ages of nodes in the ARG to be on the correct timescale, missing data
must be taken into account.  For example, if one portion of the sequence has a
large amount of missingness, an unadjusted molecular clock will tell us that it
contains substantially younger ancestry than other parts of the sequence with
less missing data. Hence, one way to account for missing data is to locally
adjust the mutation rate.

There are two sorts of missingness that are relevant: inaccessible intervals
(e.g. highly repetitive regions where variants cannot be called) and filtered
variants (e.g. "real" variants that are removed for one reason or another, such
as absence of ancestral state, missing genotypes, etc.). This pipeline
adjusts the global mutation rate within each chunk, 
```
mutation_rate[chunk] = (
  mutation_rate * 
  (accessible_sequence_length[chunk] / total_sequence_length[chunk]) * 
  (retained_variants[chunk] / total_variants[chunk])
)
``` 
where the last term is calculated considering only those variants in accessible
intervals.

An important point is that, if branch statistics are to be calculated from the
resulting ARG (e.g. `mode='branch'` in ``tskit``'s stats methods), then these
must be weighted by the adjusted mutation rate in each window and then summed across
windows to get the correct values. Some examples of this sort of calculation
can be seen in `workflow/scripts/tree_statistics.py`. The adjusted mutation
rates are saved as an ``msprime.RateMap`` object (see below).


### Outputs

The output files for each chromosome will be generated in `results/<chromosome_name>`:

  - __\<chromosome_name>.adjusted_mu.p__ : `msprime.RateMap` containing adjusted mutation rates in each chunk (see description above)
  - __\<chromosome_name>.vcf.stats.p__ : "observed values" for summary statistics (e.g. calculated from with `scikit-allel`)
  - __\<chromosome_name>.vcf__ : filtered VCF used as input to SINGER
  - __chunks/*__ the raw SINGER output and logs
  - __plots/pair-coalescence-pdf.png__, __plots/cross-coalescence-pdf.png__: pair coalescence time distribution for all samples and between strata (if supplied), with a thin line for each MCMC replicate and a thick line for the posterior mean
  - __plots/pair-coalescence-rates.png__, __plots/cross-coalescence-rates.png__: pair coalescence rates for all samples and between strata (if supplied), with a thin line for each MCMC replicate and a thick line for the posterior mean
  - __plots/diversity-trace.png__, __plots/tajima-d-trace.png__ : MCMC trace for fitted nucleotide diversity and Tajima's D
  - __plots/diversity-scatter.png__, __plots/tajima-d-scatter.png__ : observed vs fitted nucleotide diversity and Tajima's D, across chunks
  - __plots/diversity-skyline.png__, __plots/tajima-d-skyline.png__ : observed and fitted nucleotide diversity and Tajima's D, across genome position
  - __plots/folded-afs.png__, __plots/unfolded-afs.png__ : observed vs fitted site frequency spectra
  - __plots/site-density.png__ : sanity check showing proportion of missing data, proportion variant bases (out of accessible bases), average recombination rate across chunks
  - __stats/\<chromosome_name>.\<replicate>.stats.p__ : "fitted values" for summary statistics (e.g. branch-mode statistics calculated with tskit) in each chunk
  - __stats/\<chromosome_name>.\<replicate>.coalrate.p__ : pair coalescence rates (e.g. inverse of haploid Ne) within logarithmic time bins, using all samples
  - __stats/\<chromosome_name>.\<replicate>.crossrate.p__ : cross coalescence rates within logarithmic time bins, between and within strata (e.g. populations) according to the `stratify-by` option in the config file
  - __trees/\<chromosome_name>.\<replicate>.tsz__ : a tree sequence MCMC replicate generated by SINGER, compressed by ``tszip`` (e.g. use ``tszip.decompress`` to load)
