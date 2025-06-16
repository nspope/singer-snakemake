This is a Snakemake workflow for running
[SINGER](https://github.com/popgenmethods/singer) (MCMC sampling of ancestral
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

  - __chromosome_name.vcf.gz__ gzip'd VCF that can be used as SINGER input, either diploid and **phased** or haploid with an even number of samples. These should be an unbiased sample of segregating variation in the population (no ascertainment).
  - __chromosome_name.mask.bed__ (optional) bed file containing inaccessible intervals. These are intervals that a priori could not be sequenced reliably. Post-hoc filtered SNPs should go in a separate file, described below.
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
max-missing: 0.5 # ignore chunks with more than this proportion of missing bases
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

### Missing data, mutation rate, and dating ancestors

For the ages of nodes in the ARG to be on the correct timescale, missing data
must be taken into account.  For example, if the sequence has a large amount of
missingness, an unadjusted molecular clock will result in estimates of node
(ancestor) ages that are substantially younger than the truth (because fewer
mutations $\implies$ less branch area). Hence, one way to account for missing
data is to locally adjust the mutation rate.

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

This sort of crude adjustment works best when bases/variants are missing
completely at random (e.g. not too clustered along the sequence). For instance,
the figure below compares the average mutation ages between the true ARG (for
the data in `example/*`), and inferred ARGs where: (A) there is
no missing data; (B) around 60% of the sequence is masked in intervals of
average length 800bp, but the mutation rate is not adjusted; (C) the mutation
rate is adjusted to account for missing data using the scheme above.

<img src="resources/figure/missing-data-example.png" width="70%">

An important consequence is that when calculating expectations of (linear) site
statistics from branch statistics in the ARG (e.g. `mode='site'` and
`mode='branch'` in ``tskit``'s stats methods), then the contribution from each
window must be weighted by the adjusted mutation rate and then summed across
windows to get the correct values. Some examples of this sort of calculation
can be seen in `workflow/scripts/tree_statistics.py`. The adjusted mutation
rates are saved as an ``msprime.RateMap`` object (see below).

Similarly, to calculate sequence-wide distributions of coalescence time or
other topological statistics, the contribution of each window should be
weighted by the proportion of accessible sequence.


### Outputs

The output files for each chromosome will be generated in `results/<chromosome_name>`:

  - __\<chromosome_name>.adjusted_mu.p__ : `msprime.RateMap` containing adjusted mutation rates in each chunk (see description above)
  - __\<chromosome_name>.inaccessible.p__ : `msprime.RateMap` containing proportion of inaccessible bases in each chunk (see description above)
  - __\<chromosome_name>.filtered.p__ : `msprime.RateMap` containing proportion of filtered variants in each chunk (see description above)
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


### Sanity checking

To check that the timescale is correctly calibrated on synthetic data given
some missing data, it can be useful to compare the simulated (true) ARG to that
inferred by SINGER. The script `resources/scripts/compare-vs-simulation.py`
generates some visual comparisons; these include distributions of pair
coalescence times, root heights, and mutation ages of different frequencies
between the two ARGs. E.g.  after running the example workflow,

```bash
python resources/scripts/compare-vs-simulation.py \
  --true-arg "example/example.tsz" \
  --inferred-arg "results/example/trees/example.99.tsz" \
  --inaccessible-ratemap "results/example/example.inaccessible.p" \
  --output-dir "results/example/sanity-checks"
```
