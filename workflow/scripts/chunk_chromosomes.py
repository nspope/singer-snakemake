"""
Chunk chromosomes and adjust SINGER input parameters.

Part of https://github.com/nspope/singer-snakemake.
"""

import csv
import os
import msprime
import numpy as np
import allel
import matplotlib.pyplot as plt
import yaml
import pickle
import warnings
from collections import defaultdict
from datetime import datetime

warnings.simplefilter('ignore')


# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


def write_minimal_vcf(handle, sample_names, CHROM, POS, ID, REF, ALT, GT): 
    """
    Write a minimal biallelic diploid VCF
    """
    assert CHROM.size == POS.size == ID.size == REF.size
    assert ALT.ndim == 1 and ALT.size == CHROM.size
    assert GT.shape[0] == CHROM.size and GT.shape[1] == sample_names.size and GT.shape[2] == 2
    assert np.all(np.diff(POS) > 0), "Positions non-increasing in VCF"
    handle.write("##fileformat=VCFv4.2\n")
    handle.write("##source=\"singer-snakemake::chunk_chromosomes\"\n")
    handle.write("##FILTER=<ID=PASS,Description=\"All filters passed\">\n")
    handle.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
    handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
    for sample in sample_names: handle.write(f"\t{sample}")
    handle.write("\n")
    for chrom, pos, id, ref, alt, gt in zip(CHROM, POS, ID, REF, ALT, GT):
        handle.write(f"{chrom}\t{pos}\t{id}\t{ref}\t{alt}\t.\tPASS\t.\tGT")
        for (a, b) in gt: handle.write(f"\t{a}|{b}")
        handle.write("\n")


# --- implm --- #

logfile = open(snakemake.log.log, "w")
logfile.write(f"{tag()} Random seed {snakemake.params.seed}\n")
np.random.seed(snakemake.params.seed)
stratify = snakemake.params.stratify

vcf_file = snakemake.input.vcf
vcf = allel.read_vcf(vcf_file)
logfile.write(f"{tag()} Read {vcf['variants/POS'].size} variants and {vcf['samples'].size} samples from {vcf_file}\n")


# recombination map
hapmap_file = vcf_file.replace(".vcf.gz", ".hapmap")
recombination_rate = snakemake.params.recombination_rate
if not os.path.exists(hapmap_file):
    logfile.write(
        f"{tag()} Did not find {hapmap_file}, using default recombination "
        f"rate of {recombination_rate}\n"
    )
    hapmap = msprime.RateMap(
        position=np.array([0.0, np.max(vcf['variants/POS']) + 1.0]),
        rate=np.array([recombination_rate]),
    )
else:
    hapmap = msprime.RateMap.read_hapmap(hapmap_file)
    logfile.write(f"{tag()} Read {hapmap.rate.size} recombination rates from {hapmap_file}\n")


# inaccessible intervals mask (bed intervals)
mask_file = vcf_file.replace(".vcf.gz", ".mask.bed")
if not os.path.exists(mask_file):
    logfile.write(f"{tag()} Did not find {mask_file}, assuming the entire sequence is accessible\n")
    bedmask = np.empty((0, 2))
else:
    bedmask = np.loadtxt(mask_file, usecols=[1, 2]).astype(np.int64)
    assert np.max(bedmask) <= hapmap.sequence_length, "Mask position exceeds hapmap length"
    logfile.write(f"{tag()} Read {bedmask.shape[0]} inaccessible intervals from {mask_file}\n")
bitmask = np.full(int(hapmap.sequence_length), False)
for (a, b) in bedmask: bitmask[a:b] = True


# filtered sites mask (list of 1-based positions)
filter_file = vcf_file.replace(".vcf.gz", ".filter.txt")
if not os.path.exists(mask_file):
    logfile.write(f"{tag()} Did not find {filter_file}, will not adjust mutation rate for filtered SNPs\n")
    filtered_positions = np.empty(0, dtype=np.int64)
else:
    filtered_positions = np.loadtxt(filter_file).astype(np.int64)
    assert np.max(filtered_positions) <= hapmap.sequence_length, "Filtered position exceeds hapmap length"
    filtered_positions = np.unique(filtered_positions)
    logfile.write(f"{tag()} List of filtered positions includes {filtered_positions.size} variants\n")
sitemask = np.full(int(hapmap.sequence_length), False)
sitemask[filtered_positions - 1] = True


# metadata
meta_file = vcf_file.replace(".vcf.gz", ".meta.csv")
if not os.path.exists(meta_file):
    logfile.write(f"{tag()} Did not find {meta_file}, inserting sample names into metadata\n")
    metadata = [{"id":name} for name in vcf["samples"]]
    assert stratify is None, \
        f"'{meta_file}' not found, cannot stratify statistics or sample nodes " \
        f"by column '{stratify}', set to None in configfile"
else:
    meta_file = csv.reader(open(meta_file, "r"))
    metadata = []
    metadata_names = next(meta_file)
    for row in meta_file:
        assert len(row) == len(metadata_names)
        metadata.append({k:v for k, v in zip(metadata_names, row)})
    assert len(metadata) == vcf["samples"].size, "Must have a metadata row for each sample"
    if stratify is not None:
        assert stratify in metadata_names, \
            f"Cannot stratify statistics or sample nodes by column " \
            f"'{stratify}' that isn't in the metadata"


# convert to diploid VCF if necessary
assert vcf['calldata/GT'].shape[2] == 2
ploidy = 1 if np.all(vcf['calldata/GT'][..., 1] == -1) else 2
if ploidy == 1:
    assert vcf['samples'].size % 2 == 0, "VCF is haploid with an odd number of samples: cannot diploidize"
    logfile.write(f"{tag()} VCF is haploid, converting to diploid for SINGER\n")
    vcf['samples'] = np.array([f"{a}_{b}" for a, b in zip(vcf['samples'][::2], vcf['samples'][1::2])])
    vcf['calldata/GT'] = vcf['calldata/GT'][..., 0].reshape(-1, vcf['samples'].size, 2)
samples = vcf['samples']
genotypes = allel.GenotypeArray(vcf['calldata/GT']) 
positions = vcf['variants/POS']
assert np.max(positions) <= hapmap.sequence_length, "VCF position exceeds hapmap length"


# NB: to adjust mutation rate to account for missing data, there are two relevant
# types of missingness: inaccessible intervals and filtered sites. The adjustment
# has the form,
#
# `adjusted_mu = mu * (filtered_intervals / sequence_length) * (filtered_sites / segregating_sites)`
#
# where the last term is calculated *discounting* sites that lie in filtered
# intervals.


# filter variants
counts = genotypes.count_alleles()
filter_segregating = counts.is_segregating()
logfile.write(f"{tag()} Removed {np.sum(~filter_segregating)} non-segregating sites\n")
filter_biallelic = counts.is_biallelic()
logfile.write(f"{tag()} Removed {np.sum(~filter_biallelic)} non-biallelic sites\n")
filter_nonmissing = counts.sum(axis=1) == 2 * samples.size
logfile.write(f"{tag()} Removed {np.sum(~filter_nonmissing)} sites with missing data\n")
filter_accessible = ~bitmask[positions - 1]
logfile.write(f"{tag()} Removed {np.sum(~filter_accessible)} sites occuring in inaccessible intervals\n")
filter_unmasked = ~sitemask[positions - 1]
logfile.write(f"{tag()} Removed {np.sum(~filter_unmasked)} sites explicitly marked as filtered SNPs\n")
retain = np.logical_and.reduce([
    filter_segregating, 
    filter_biallelic, 
    filter_nonmissing, 
    filter_accessible,
    filter_unmasked,
])


# update list of filtered sites to remove inaccessible, but include
# those that are multiallelic or contain missing data
filtered_positions = np.concatenate([
    filtered_positions, 
    positions[np.logical_and(filter_segregating, ~filter_nonmissing)],
    positions[np.logical_and(filter_segregating, ~filter_biallelic)],
])
filtered_positions = np.unique(filtered_positions)
logfile.write(
    f"{tag()} Added multiallelic and variants with missing genotypes "
    f"to the list of filtered variants (size {filtered_positions.size})\n"
)
filtered_positions = filtered_positions[~bitmask[filtered_positions - 1]]
logfile.write(
    f"{tag()} Removed inaccessible positions from the list "
    f"of filtered variants, left with {filtered_positions.size} variants\n"
)
assert np.all(~bitmask[filtered_positions - 1])


# apply filters to genotypes, positions
assert np.sum(retain) > 0, "No variants left after filtering"
logfile.write(f"{tag()} Calculating statistics with remaining {np.sum(retain)} variants\n")
genotypes, positions = genotypes[retain], positions[retain]
counts = genotypes.count_alleles(max_allele=1)
assert counts.shape[1] == 2


# divvy chromosome into chunks
chunk_size = snakemake.params.chunk_size
mutation_rate = snakemake.params.mutation_rate
lower = np.min(positions) - 1 
upper = np.max(positions) + 1 
windows = np.linspace(lower, upper, int((upper - lower) / chunk_size) + 1)
windows = np.unique(np.append(np.append(0, windows), hapmap.sequence_length))
windows = windows.astype(np.int64)
logfile.write(
    f"{tag()} Chunking chromosome into {windows.size - 1} windows "
    f"from {windows[0]} to {windows[-1]}\n"
)


# average recombination rate within chunks
rec_rate = np.diff(hapmap.get_cumulative_mass(windows)) / np.diff(windows)


# tally filtered sequence and variants per window
num_bases = np.diff(windows)
num_accessible = np.array([  # total length of unmasked intervals
    sum(~bitmask[a:b]) for a, b 
    in zip(windows[:-1], windows[1:])
])
logfile.write(f"{tag()} Counted {sum(num_accessible)} accessible out of {sum(num_bases)} total bases\n")
num_retained = np.bincount(  # sites passing all filters
    np.digitize(positions - 1, windows) - 1,
    minlength=windows.size - 1,
)
num_filtered = np.bincount(  # accessible but filtered sites
    np.digitize(filtered_positions - 1, windows) - 1,
    minlength=windows.size - 1,
)
assert num_filtered.size == num_retained.size == windows.size - 1
logfile.write(f"{tag()} Counted {sum(num_retained)} retained and {sum(num_filtered)} filtered variants\n")


## TODO: ultimately remove this, use calculations above
## count missing bases
#*_, num_nonmissing, num_sites = allel.windowed_diversity(
#    positions,
#    counts,
#    windows=np.column_stack([windows[:-1] + 1, windows[1:]]),
#    is_accessible=~bitmask,
#    fill=0.0,
#)


# filter chunks with too much missingness or zero recombination rate
# TODO: delete
#num_total = np.diff(windows)
#num_missing = num_total - num_nonmissing
#prop_missing = num_missing / num_total
#prop_snp = num_sites / num_nonmissing
#prop_snp[np.isnan(prop_snp)] = 0.0
#filter_chunks = np.logical_and(prop_missing < snakemake.params.max_missing, prop_snp > 0.0)
#filter_chunks = np.logical_and(filter_chunks, rec_rate > 0.0)
#logfile.write(
#    f"{tag()} Skipping {np.sum(~filter_chunks)} (of {filter_chunks.size}) "
#    f"chunks with too much missing data or zero recombination rate\n"
#)
prop_inaccessible = (num_bases - num_accessible) / num_bases  #<< prop_missing
prop_segregating = num_retained / num_accessible  #<< prop_snp
prop_segregating[np.isnan(prop_segregating)] = 0.0
prop_filtered = num_filtered / (num_retained + num_filtered)  #<<<new
prop_filtered[np.isnan(prop_filtered)] = 1.0
filter_chunks = np.logical_and.reduce([
    prop_inaccessible < snakemake.params.max_missing, 
    prop_filtered < snakemake.params.max_missing,
    prop_segregating > 0.0,
    rec_rate > 0.0,
])
logfile.write(
    f"{tag()} Skipping {np.sum(~filter_chunks)} (of {filter_chunks.size}) "
    f"chunks with too much missing data or zero recombination rate\n"
)


# update site mask to reflect filtered chunks (used for global statistics calculation)
for i in np.flatnonzero(~filter_chunks):
    start, end = windows[i], windows[i + 1]
    bitmask[start:end] = True
logfile.write(
    f"{tag()} Updating sequence mask with skipped chunks, went from "
    f"{np.sum(num_accessible)} to {np.sum(~bitmask)} unmasked bases\n"
)
filtered_positions = filtered_positions[~bitmask[filtered_positions - 1]]
logfile.write(
    f"{tag()} Updating list of filtered positions with skipped chunks, went from "
    f"{np.sum(num_filtered)} to {filtered_positions.size} filtered variants\n"
)
num_accessible[~filter_chunks] = 0
num_retained[~filter_chunks] = 0
num_filtered[~filter_chunks] = 0
assert num_bases.sum() == bitmask.size
assert num_accessible.sum() == np.sum(~bitmask)
assert num_filtered.sum() == filtered_positions.size
assert num_retained.sum() == positions.size == np.sum(retain)


# calculate windowed stats and get ballpark Ne estimate from global diversity
diversity, *_ = allel.windowed_diversity(
    positions,
    counts,
    windows=np.column_stack([windows[:-1] + 1, windows[1:]]),
    is_accessible=~bitmask,
    fill=0.0,
)
tajima_d, *_ = allel.windowed_tajima_d(
    positions,
    counts,
    windows=np.column_stack([windows[:-1] + 1, windows[1:]]),
)
folded_afs = allel.sfs_folded(counts, n=2 * samples.size) / np.sum(~bitmask)
unfolded_afs = allel.sfs(counts[:, 1], n=2 * samples.size) / np.sum(~bitmask)
adjustment = sum(num_retained) / sum(num_retained + num_filtered)
Ne = (
    allel.sequence_diversity(positions, counts, is_accessible=~bitmask) *
    1 / (4 * mutation_rate * adjustment)
)
logfile.write(
    f"{tag()} Using ballpark Ne estimate of {Ne} after scaling mutation "
    f"rate by factor of {adjustment:.3f} to account for filtered sites\n"
)


# plot site density and recombination rate as sanity check
fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
col = ['black' if x else 'red' for x in filter_chunks]
for l, r in zip(windows[:-1][~filter_chunks], windows[1:][~filter_chunks]):
    for ax in axs: 
        ax.axvspan(l, r, edgecolor=None, facecolor='firebrick', alpha=0.1)
        ax.set_xlim(windows[0], windows[-1])
axs[0].step(windows[:-1], prop_inaccessible, where="post", color="black")
axs[0].set_ylabel("Proportion inaccessible")
axs[1].step(windows[:-1], prop_segregating, where="post", color="black")
axs[1].set_ylabel("Proportion variant bases")
axs[2].step(windows[:-1], prop_filtered, where="post", color="black")
axs[2].set_ylabel("Proportion filtered")
axs[3].step(windows[:-1], rec_rate, where="post", color="black")
axs[3].set_ylabel("Recombination rate")
axs[3].set_yscale("log")
fig.supxlabel("Position")
fig.tight_layout()
plt.savefig(snakemake.output.site_density)
plt.clf()


# adjust mutation rate to account for missing data in each chunk
chunks_dir = snakemake.output.chunks
os.makedirs(f"{chunks_dir}")
seeds = np.random.randint(0, 2**10, size=filter_chunks.size)
vcf_prefix = snakemake.output.vcf.removesuffix(".vcf")
adj_mu = np.zeros(windows.size - 1)
for i in np.flatnonzero(filter_chunks):
    start, end = windows[i], windows[i + 1]
    adj_mu[i] = (1 - prop_inaccessible[i]) * (1 - prop_filtered[i]) * mutation_rate
    polar = 0.99 if snakemake.params.polarised else 0.5
    id = f"{i:>06}"
    chunk_params = {
        "thin": int(snakemake.params.mcmc_thin), 
        "n": int(snakemake.params.mcmc_samples),
        "Ne": float(Ne),
        "m": float(adj_mu[i]), 
        "input": str(vcf_prefix), 
        "start": int(start), 
        "end": int(end), 
        "polar": float(polar),
        "r": float(rec_rate[i]), 
        "seed": int(seeds[i]),
        "output": str(f"{chunks_dir}/{id}"),
    }
    chunk_path = f"{chunks_dir}/{id}.yaml"
    yaml.dump(chunk_params, open(chunk_path, "w"), default_flow_style=False)
    logfile.write(f"{tag()} Parameters for chunk {id} are {chunk_params}\n")


# dump adjusted mutation rates and chunk coordinates
ratemap = msprime.RateMap(
    position=windows, 
    rate=np.array(adj_mu),
)
pickle.dump(ratemap, open(snakemake.output.ratemap, "wb"))


# dump filtered vcf
write_minimal_vcf(
    open(snakemake.output.vcf, "w"),
    vcf['samples'],
    vcf['variants/CHROM'][retain],
    vcf['variants/POS'][retain],
    vcf['variants/ID'][retain],
    vcf['variants/REF'][retain],
    vcf['variants/ALT'][retain, 0],
    vcf['calldata/GT'][retain],
)


# dump statistics
diversity[~filter_chunks] = np.nan
tajima_d[~filter_chunks] = np.nan
vcf_stats = {
    "diversity": diversity, 
    "tajima_d": tajima_d, 
    "folded_afs": folded_afs, 
    "unfolded_afs": unfolded_afs,
}
pickle.dump(vcf_stats, open(snakemake.output.vcf_stats, "wb"))
pickle.dump(metadata, open(snakemake.output.metadata, "wb"))


# TODO: clean this up
# stratified summary stats
vcf_strata_stats = {}
if stratify is not None:
    sample_sets = defaultdict(list)
    for i, md in enumerate(metadata):
        sample_sets[md[stratify]].append(i)
    strata = [n for n in sample_sets.keys()]
    strata_counts = genotypes.count_alleles_subpops(sample_sets, max_allele=1)

    strata_divergence = []
    strata_folded_afs = []
    strata_unfolded_afs = []
    for i in range(len(strata)):
        for j in range(i, len(strata)):
            divergence, *_ = allel.windowed_divergence(
                positions,
                strata_counts[strata[i]],
                strata_counts[strata[j]],
                windows=np.column_stack([windows[:-1] + 1, windows[1:]]),
                is_accessible=~bitmask,
                fill=0.0,
            )
            strata_divergence.append(divergence)

        folded_afs = allel.sfs_folded(
            strata_counts[strata[i]], 
            n=2 * len(sample_sets[strata[i]]),
        ) / np.sum(~bitmask)
        strata_folded_afs.append(folded_afs)

        unfolded_afs = allel.sfs(
            strata_counts[strata[i]][:, 1], 
            n=2 * len(sample_sets[strata[i]]),
        ) / np.sum(~bitmask)
        strata_unfolded_afs.append(unfolded_afs)

    strata_divergence = np.stack(strata_divergence).T
    vcf_strata_stats = {
        "strata": strata,
        "divergence": strata_divergence,
        "folded_afs": strata_folded_afs,
        "unfolded_afs": strata_unfolded_afs,
    }
pickle.dump(vcf_strata_stats, open(snakemake.output.vcf_strata_stats, "wb"))
