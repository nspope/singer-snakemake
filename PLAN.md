# Scope

Adding handling of per-sample missing data. The idea is to impute genotypes over intervals for SINGER to build trees with,
but then to prune out ancestry partially for dating. The output tree sequences will have trees with variable numbers of
samples.

# Snakefile

Minor changes.

- Pass in per sample mask to chunk chroms as bedfile with extra column
  indicating sample, pass out pickled dict with intervals.
  Output gzipped vcf that "process_missing" rule will act on.
- New rule for "process_missing.py". Output uncompressed vcf that downstream rules will act on.
- Rewire outputs/inputs for run_polegon.py and merge_chunks.py according to discussion below.
- New rule for calculating observed stats.
- "plot diagonostics" rule will take "observed from tree sequence" rather than "observed from vcf"

# chunk_chromosomes.py

Major changes.

- Process per sample masks, put the intervals in a dict
- Any intervals with >= max_missing get added to the global mask
- Drop all snps in globally masked regions
- Set genotypes to missing in masked per-sample intervals
- Don't drop sites with partially missing data
- Don't calculate stats aside from those needed for filtering/Ne
- Make sure repolarization does something sane when there are missing genotypes
- Output gzip'd VCF

# process_missing.py

New file. This is responsible for filling in missing genotypes so that sites are not dropped by SINGER.

- By default, drop sites with any missing data
- Otherwise, use frequency to impute (meaning: sampling with replacement from nonmissing genotype vector to fill out missing genotypes)
- Output uncompressed VCF

# run_polegon.py

Major changes.

- Convert singer output to tree sequence
- prune ancestry, remap coordinates, output *all* polegon inputs (we're already doing this for naming convention)

Then,

OPTION A: Output tree sequence

or

OPTION B: Output singer-style flat outputs. The big question is to what degree these should reflect the pruning of ancestry.
  There's a couple options here.

   - (B1): Work in the original ARG topology. Here, we just change edge spans / mutation counts so as to get the correct node ages.
      Some of these nodes will ultimately be removed (because they only subtend samples over "missing intervals" for these samples).
      In other words, (1) we adjust edge spans to account for pruned ancestry, which likely involves creating a new pruned tree sequence and
      comparing to the original to get the edge map, (2) we simply output a new set of node ages as before, which includes "pruned" nodes.

   - (B2): Work with the pruned ARG topology. Here, we do the pruning in run_polegon, date the pruned tree sequence, and output multiple
     files from the run_polegon step with the new topology. This could break downstream stuff but avoids having to re-prune later on.

# merge_chunks.py

No changes or major changes:

OPTION A (major changes):

- Concatenate tree sequences, but need a way to have offset/gap between chunks
- Metadata would need to be added as well.

OPTION B (no changes):

- Act on inputs of the same form (flat SINGER-style text files). The big question is whether "missing ancestry" interferes with the current machinery.
- To keep things as close as possible to the current execution path, we can avoid doing the pruning of missing ancestry in the
  run polegon step. In other words, we just update edge spans for dating, but don't change the arg topology. Then, we need a new
  step in merge_chunks to actually prune the ancestry. These corresponds to (B1) above.

# tree_statistics.py

Minor changes.

- Add "use observed" flag that doesn't sim mutations, this will then output the "observed" site statistics for an MCMC rep.

# plot_diagnostics.py

Minor changes.

- Will need to average over observed site stats from each rep.


# Infrastructure needs

- Utility to remove partial ancestry from a tree sequence
- Utility to manage intervals (clip and shift, intersect, etc)
- Utilities for reading/writing tree sequences for POLEGON

These should be added to utils.py and unit tested.
