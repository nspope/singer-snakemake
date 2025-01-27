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
from collections import defaultdict
from datetime import datetime

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

# recombination map
hapmap_file = vcf_file.replace(".vcf.gz", ".hapmap")
recombination_rate = snakemake.params.recombination_rate
if not os.path.exists(hapmap_file):
    logfile.write(
        f"{tag()} Did not find {hapmap_file}, using default recombination "
        f"rate of {recombination_rate}\n")
    hapmap = msprime.RateMap(
        position=np.array([0.0, np.max(vcf['variants/POS']) + 1.0]),
        rate=np.array([recombination_rate]),
    )
else:
    hapmap = msprime.RateMap.read_hapmap(hapmap_file)

# missing data mask
mask_file = vcf_file.replace(".vcf.gz", ".mask.bed")
if not os.path.exists(mask_file):
    logfile.write(f"{tag()} Did not find {mask_file}, assuming the entire sequence is accessible\n")
    bedmask = np.empty((0, 2))
else:
    bedmask = np.loadtxt(mask_file, usecols=[1, 2]).astype(np.int64)
    assert np.max(bedmask) <= hapmap.sequence_length, "Mask position exceeds hapmap length"
bitmask = np.full(int(hapmap.sequence_length), False)
for (a, b) in bedmask: bitmask[a:b] = True

# metadata
meta_file = vcf_file.replace(".vcf.gz", ".meta.csv")
if not os.path.exists(meta_file):
    logfile.write(f"{tag()} Did not find {meta_file}, inserting sample names into metadata\n")
    metadata = [{"id":name} for name in vcf["samples"]]
    assert stratify is None, f"\"{meta_file}\" not found, cannot stratify statistics by column \"{stratify}\", use None instead"
else:
    meta_file = csv.reader(open(meta_file, "r"))
    metadata = []
    metadata_names = next(meta_file)
    for row in meta_file:
        assert len(row) == len(metadata_names)
        metadata.append({k:v for k, v in zip(metadata_names, row)})
    assert len(metadata) == vcf["samples"].size, "Must have a metadata row for each sample"
    if stratify is not None:
        assert stratify in metadata_names, f"Cannot stratify statistics by column \"{stratify}\" that isn't in metadata"

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

# filter variants to biallelic, nonmasked
counts = genotypes.count_alleles()
filter_biallelic = np.logical_and(counts.is_segregating(), counts.is_biallelic())
logfile.write(f"{tag()} Removed {np.sum(~filter_biallelic)} non-biallelic sites\n")
filter_missing = counts.sum(axis=1) == 2 * samples.size
logfile.write(f"{tag()} Removed {np.sum(~filter_missing)} sites with missing data\n")
masked_sites = np.sum(bitmask[positions - 1])
logfile.write(f"{tag()} Removed {masked_sites} sites occuring in masked regions\n")

retain = np.logical_and(filter_biallelic, filter_missing)
retain = np.logical_and(retain, ~bitmask[positions - 1])
assert np.sum(retain) > 0, "No variants left after filtering"
logfile.write(f"{tag()} Calculating statistics with remaining {np.sum(retain)} variants\n")
genotypes, positions = genotypes[retain], positions[retain]
counts = genotypes.count_alleles(max_allele=1)
assert counts.shape[1] == 2

# divvy chromosome into chunks and compute summary stats per chunk
chunk_size = snakemake.params.chunk_size
mutation_rate = snakemake.params.mutation_rate
lower = np.min(positions) - 1 
upper = np.max(positions) + 1 
windows = np.linspace(lower, upper, int((upper - lower) / chunk_size) + 1)
windows = np.unique(np.append(np.append(0, windows), hapmap.sequence_length))
logfile.write(
    f"{tag()} Chunking chromosome into {windows.size - 1} windows "
    f"from {windows[0]} to {windows[-1]}\n"
)

# count missing bases
*_, num_nonmissing, num_sites = allel.windowed_diversity(
    positions,
    counts,
    windows=np.column_stack([windows[:-1] + 1, windows[1:]]).astype(np.int64),
    is_accessible=~bitmask,
    fill=0.0,
)

# average recombination rate within chunks
rec_rate = np.diff(hapmap.get_cumulative_mass(windows)) / np.diff(windows)

# filter chunks with too much missingness or zero recombination rate
num_total = np.diff(windows)
num_missing = num_total - num_nonmissing
prop_missing = num_missing / num_total
prop_snp = num_sites / num_nonmissing
prop_snp[np.isnan(prop_snp)] = 0.0
filter = np.logical_and(prop_missing < snakemake.params.max_missing, prop_snp > 0.0)
filter = np.logical_and(filter, rec_rate > 0.0)
logfile.write(
    f"{tag()} Skipping {np.sum(~filter)} (of {filter.size}) "
    f"chunks with too much missing data or zero recombination rate\n"
)

# update site mask to reflect filtered chunks
for i in np.flatnonzero(~filter):
    start, end = windows[i], windows[i + 1]
    bitmask[int(start):int(end)] = True
logfile.write(
    f"{tag()} Updating site mask with skipped chunks, went from "
    f"{np.sum(num_nonmissing)} to {np.sum(~bitmask)} unmasked bases\n"
)

# calculate stats and get ballpark Ne estimate
diversity, *_ = allel.windowed_diversity(
    positions,
    counts,
    windows=np.column_stack([windows[:-1] + 1, windows[1:]]).astype(np.int64),
    is_accessible=~bitmask,
    fill=0.0,
)
tajima_d, *_ = allel.windowed_tajima_d(
    positions,
    counts,
    windows=np.column_stack([windows[:-1] + 1, windows[1:]]).astype(np.int64),
)
folded_afs = allel.sfs_folded(counts, n=2 * samples.size) / np.sum(~bitmask)
unfolded_afs = allel.sfs(counts[:, 1], n=2 * samples.size) / np.sum(~bitmask)
Ne = 0.25 * allel.sequence_diversity(positions, counts, is_accessible=~bitmask) / mutation_rate
logfile.write(f"{tag()} Using ballpark Ne estimate of {Ne}\n")

# plot site density and recombination rate as sanity check
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
col = ['black' if x else 'red' for x in filter]
axs[0].scatter(windows[:-1], prop_missing, s=4, c=col)
axs[0].set_ylabel("Proportion missing bases")
axs[1].scatter(windows[:-1], prop_snp, s=4, c=col)
axs[1].set_ylabel("Proportion variant bases")
axs[2].scatter(windows[:-1], rec_rate, s=4, c=col)
axs[2].set_ylabel("Recombination rate")
axs[2].set_yscale("log")
fig.supxlabel("Position")
fig.tight_layout()
plt.savefig(snakemake.output.site_density)
plt.clf()

# adjust mutation rate to account for missing data in each chunk
chunks_dir = snakemake.output.chunks
os.makedirs(f"{chunks_dir}")
seeds = np.random.randint(0, 2**10, size=filter.size)
vcf_prefix = snakemake.output.vcf.removesuffix(".vcf")
adj_mu = np.zeros(windows.size - 1)
for i in np.flatnonzero(filter):
    start, end = windows[i], windows[i + 1]
    adj_mu[i] = (1 - prop_missing[i]) * mutation_rate
    polar = 0.99 if snakemake.params.polarised else 0.5
    id = f"{i:>06}"
    chunk_params = {
        "thin" : int(snakemake.params.mcmc_thin), 
        "n" : int(snakemake.params.mcmc_samples),
        "Ne" : float(Ne),
        "m" : float(adj_mu[i]), 
        "input" : str(vcf_prefix), 
        "start" : int(start), 
        "end" : int(end), 
        "polar" : float(polar),
        "r" : float(rec_rate[i]), 
        "seed" : int(seeds[i]),
        "output" : str(f"{chunks_dir}/{id}"),
    }
    chunk_path = f"{chunks_dir}/{id}.yaml"
    yaml.dump(chunk_params, open(chunk_path, "w"), default_flow_style=False)

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
diversity[~filter] = np.nan
tajima_d[~filter] = np.nan
vcf_stats = {
    "diversity" : diversity, 
    "tajima_d" : tajima_d, 
    "folded_afs" : folded_afs, 
    "unfolded_afs" : unfolded_afs,
}
pickle.dump(vcf_stats, open(snakemake.output.vcf_stats, "wb"))
pickle.dump(metadata, open(snakemake.output.metadata, "wb"))

# stratified summary stats (TODO)
#if stratify is not None:
#    sample_sets = defaultdict(list)
#    for i, md in enumerate(metadata):
#        sample_sets[md[stratify]].append(i)
#    stratified_counts = genotypes.count_alleles_subpop(sample_sets)
#    stratified_diversity = {}
#    stratified_tajima_d = {}
#    stratified_folded_afs = {}
#    stratified_unfolded_afs = {}
#    for nm, cnt in stratified_counts.items():
#        stratified_diversity[nm] = allel.windowed_diversity(...)
#        stratified_tajima_d[nm] = allel.windowed_tajima_d(...)
#    raise ValueError("Not implemented yet")
