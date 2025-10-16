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

from utils import bitmask_to_arrays
from utils import ratemap_to_text

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
rng = np.random.default_rng(snakemake.params.seed)
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
    bedmask = np.loadtxt(mask_file, usecols=[1, 2], ndmin=2).astype(np.int64)
    assert np.max(bedmask) <= hapmap.sequence_length, "Mask position exceeds hapmap length"
    logfile.write(f"{tag()} Read {bedmask.shape[0]} inaccessible intervals from {mask_file}\n")
bitmask = np.full(int(hapmap.sequence_length), False)
for (a, b) in bedmask: bitmask[a:b] = True


# filtered sites mask (list of 1-based positions)
filter_file = vcf_file.replace(".vcf.gz", ".filter.txt")
if not os.path.exists(filter_file):
    logfile.write(f"{tag()} Did not find {filter_file}, will not adjust mutation rate for filtered SNPs\n")
    filtered_positions = np.empty(0, dtype=np.int64)
else:
    filtered_positions = np.loadtxt(filter_file).astype(np.int64)
    if filtered_positions.size:
        assert np.max(filtered_positions) <= hapmap.sequence_length, "Filtered position exceeds hapmap length"
        filtered_positions = np.unique(filtered_positions)
    logfile.write(f"{tag()} List of filtered positions includes {filtered_positions.size} variants\n")
sitemask = np.full(int(hapmap.sequence_length), False)
sitemask[filtered_positions - 1] = True


# sites omitted from dating (list of 1-based positions)
omit_file = vcf_file.replace(".vcf.gz", ".omit.txt")
if not os.path.exists(omit_file):
    logfile.write(f"{tag()} Did not find {omit_file}, will not omit specific variants from ARG dating\n")
    omitted_positions = np.empty(0, dtype=np.int64)
else:
    omitted_positions = np.loadtxt(omit_file).astype(np.int64)
    if omitted_positions.size:
        assert np.max(omitted_positions) <= hapmap.sequence_length, "Omitted position exceeds hapmap length"
        omitted_positions = np.unique(omitted_positions)
    logfile.write(
        f"{tag()} List of omitted positions includes {omitted_positions.size} variants "
        f"that will be used to build topologies but not for dating\n"
    )
datemask = np.full(int(hapmap.sequence_length), False)
datemask[omitted_positions - 1] = True


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


# filter variants
counts = genotypes.count_alleles()
filter_segregating = counts.is_segregating()
logfile.write(f"{tag()} Removed {np.sum(~filter_segregating)} non-segregating sites\n")
filter_biallelic = genotypes.max(axis=(-2, -1)) == 1
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
all_positions = positions.copy()
genotypes, positions = genotypes[retain], positions[retain]


# find inaccessible intervals greater than a certain size, and use these
# to delimit the chunks where SINGER will be run. If chunks are larger than
# a certain size, split them into equal subdivisions.
mutation_rate = snakemake.params.mutation_rate
max_chunk_size = snakemake.params.max_chunk_size
min_chunk_size = snakemake.params.min_chunk_size
min_sites = snakemake.params.min_sites
max_missing = snakemake.params.max_missing
min_gap_size = snakemake.params.min_gap_size
sequence_length = bitmask.size
lower = np.min(positions) - 1 
upper = np.max(positions) + 1 
gap_size, intervals = bitmask_to_arrays(bitmask)
gap = np.logical_and(np.diff(intervals) > min_gap_size, gap_size > 0)
intervals = intervals.astype(np.int64)
gapmask = np.full(bitmask.size, False)
gapmask[:int(lower)] = True
gapmask[int(upper):] = True
for (a, b) in zip(intervals[:-1][gap], intervals[1:][gap]):
    gapmask[a:b] = True
_, windows = bitmask_to_arrays(~gapmask)
logfile.write(
    f"{tag()} Found {gap.sum()} masked intervals larger than {int(min_gap_size)} bp "
    f"(including flanks outside of variant positions) that delimit {windows.size - 1} "
    f"regions\n"
)
windows_refined = [0.0]
for (a, b) in zip(windows[:-1], windows[1:]):
    assert a == windows_refined[-1]
    if b - a > max_chunk_size:
        subdivisions = int(np.ceil((b - a) / max_chunk_size))
        windows_refined.extend(np.linspace(a, b, subdivisions + 1)[1:])
    else:
        windows_refined.append(b)
assert windows_refined[-1] == sequence_length
windows = np.array(windows_refined).astype(np.int64)
logfile.write(
    f"{tag()} Subdividing each region with maximum subdivision size {int(max_chunk_size)} bp, "
    f"resulting in {windows.size - 1} chunks\n"
)


# average recombination rate within chunks
rec_rate = np.diff(hapmap.get_cumulative_mass(windows)) / np.diff(windows)


# tally filtered sequence and variants per window
num_bases = np.diff(windows)
num_accessible = np.array([  # total length of unmasked intervals
    sum(~bitmask[a:b]) for a, b 
    in zip(windows[:-1], windows[1:])
])
# TODO: use np.add.reduceat
#num_accessible_2 = np.add.reduceat(bitmask, windows) #TODO
#np.testing.assert_allclose(num_accessible, num_accessible_2)
#/TODO
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


# filter chunks with too much missingness or zero recombination rate
prop_inaccessible = (num_bases - num_accessible) / num_bases
prop_segregating = num_retained / num_accessible
prop_segregating[np.isnan(prop_segregating)] = 0.0
prop_filtered = num_filtered / (num_retained + num_filtered)
prop_filtered[np.isnan(prop_filtered)] = 1.0
filter_min_size = num_bases >= min_chunk_size
logfile.write(
    f"{tag()} Skipping {np.sum(~filter_min_size)} chunks smaller than "
    f"{int(min_chunk_size)} bp\n"
)
filter_inaccessible = prop_inaccessible <= max_missing
logfile.write(
    f"{tag()} Skipping {np.sum(~filter_inaccessible)} chunks with a proportion of "
    f"masked sequence greater than {max_missing}\n"
)
filter_filtered = prop_filtered <= max_missing
logfile.write(
    f"{tag()} Skipping {np.sum(~filter_filtered)} chunks with a proportion of "
    f"filtered variants greater than {max_missing}\n"
)
filter_min_sites = num_retained >= min_sites
logfile.write(
    f"{tag()} Skipping {np.sum(~filter_min_sites)} chunks with fewer than "
    f"{int(min_sites)} variants in unmasked intervals\n"
)
filter_recombinant = rec_rate > 0.0
logfile.write(
    f"{tag()} Skipping {np.sum(~filter_recombinant)} chunks with zero "
    f"recombination rate\n"
)
filter_chunks = np.logical_and.reduce([
    filter_min_size,
    filter_inaccessible,
    filter_filtered,
    filter_min_sites,
    filter_recombinant,
])
logfile.write(
    f"{tag()} Skipping {np.sum(~filter_chunks)} chunks "
    f"(of {filter_chunks.size} total)\n"
)


# plot site density and recombination rate as sanity check,
# filtered chunks are highlighted with red
fig, axs = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
col = ['black' if x else 'red' for x in filter_chunks]
for l, r in zip(windows[:-1][~filter_chunks], windows[1:][~filter_chunks]):
    for ax in axs: 
        ax.axvspan(l, r, edgecolor=None, facecolor='firebrick', alpha=0.1)
        ax.set_xlim(windows[0], windows[-1])
axs[0].step(windows[:-1], prop_inaccessible, where="post", color="black")
axs[0].set_ylabel("Proportion masked bases")
axs[1].step(windows[:-1], prop_segregating, where="post", color="black")
axs[1].set_ylabel("Proportion variant bases")
axs[2].step(windows[:-1], prop_filtered, where="post", color="black")
axs[2].set_ylabel("Proportion filtered variants")
axs[3].step(windows[:-1], rec_rate, where="post", color="black")
axs[3].set_ylabel("Mean recombination rate")
axs[3].set_yscale("log")
fig.supxlabel("Position")
fig.tight_layout()
plt.savefig(snakemake.output.site_density)
plt.clf()


# update site mask to reflect filtered chunks (used for global statistics calculation)
for i in np.flatnonzero(~filter_chunks):
    start, end = windows[i], windows[i + 1]
    bitmask[start:end] = True
logfile.write(
    f"{tag()} Updating sequence mask given filtered chunks, went from "
    f"{np.sum(num_accessible)} to {np.sum(~bitmask)} unmasked bases\n"
)
filtered_positions = filtered_positions[~bitmask[filtered_positions - 1]]
logfile.write(
    f"{tag()} Updating list of filtered positions given filtered chunks, went from "
    f"{np.sum(num_filtered)} to {filtered_positions.size} filtered variants\n"
)
retain[bitmask[all_positions - 1]] = False
genotypes = genotypes[~bitmask[positions - 1]]
positions = positions[~bitmask[positions - 1]]
logfile.write(
    f"{tag()} Updating list of variant positions given filtered chunks, went from "
    f"{np.sum(num_retained)} to {positions.size} filtered variants\n"
)
num_accessible[~filter_chunks] = 0
num_retained[~filter_chunks] = 0
num_filtered[~filter_chunks] = 0
prop_inaccessible[~filter_chunks] = 1.0
prop_filtered[~filter_chunks] = 0.0
prop_segregating[~filter_chunks] = 0.0
assert num_bases.sum() == bitmask.size
assert num_accessible.sum() == np.sum(~bitmask)
assert num_filtered.sum() == filtered_positions.size
assert num_retained.sum() == positions.size == np.sum(retain)
# NB: some of these arrays may be recalculated at a finer scale to adjust mutation rate


# calculate windowed stats and get ballpark Ne estimate from global diversity
num_stats_windows = int(np.floor(bitmask.size / snakemake.params.stats_window_size))
statistics_windows = np.linspace(0, bitmask.size, num_stats_windows + 1).astype(int)
statistics_windows_stack = \
    np.column_stack([statistics_windows[:-1] + 1, statistics_windows[1:]])
statmask = np.logical_and(~bitmask, ~datemask)
statkeep = statmask[positions - 1]  # statistics should not use sites omitted from dating
logfile.write(
    f"{tag()} Using {statkeep.sum()} sites to calculate observed summary statistics "
    f"(omitting {positions.size - statkeep.sum()} sites that will not be used for dating)\n"
)
counts = genotypes.count_alleles(max_allele=1)
assert counts.shape[1] == 2
diversity, *_ = allel.windowed_diversity(
    positions[statkeep],
    counts[statkeep],
    windows=statistics_windows_stack,
    is_accessible=statmask,
    fill=0.0,
)
tajima_d, *_ = allel.windowed_tajima_d(
    positions[statkeep],
    counts[statkeep],
    windows=statistics_windows_stack,
)
if snakemake.params.polarised:
    afs = allel.sfs(counts[statkeep, 1], n=2 * samples.size) / np.sum(statmask) 
else:
    afs = allel.sfs_folded(counts[statkeep], n=2 * samples.size) / np.sum(statmask)
adjustment = sum(num_retained) / sum(num_retained + num_filtered)
Ne = (
    allel.sequence_diversity(positions[statkeep], counts[statkeep], is_accessible=statmask) *
    1 / (4 * mutation_rate * adjustment)
)
logfile.write(
    f"{tag()} Using ballpark Ne estimate of {Ne:.2f} after scaling mutation "
    f"rate by factor of {adjustment:.3f} to account for filtered sites\n"
)


# adjust mutation rate to account for masked sequence in each chunk, either by
# making a rate map where masked sequence has zero rate or incorporating into
# the global mutation rate
if snakemake.params.model_masked_sequence:
    logfile.write(
        f"{tag()} Adjusting per-chunk mutation rate to account for filtered sites "
        f"and setting mutation rate to zero within masked intervals\n"
    )
    # replace `prop_inaccessible` with a per-base binary mask
    prop_inaccessible, breakpoints = bitmask_to_arrays(bitmask, insert_breakpoints=windows)
    assert breakpoints[-1] == windows[-1] and breakpoints[0] == windows[0]
    logfile.write(
        f"{tag()} Fine-scale mutation rate map has {breakpoints.size - 1} "
        f"intervals, with {int(prop_inaccessible.sum())} that are masked\n"
    )
    # map `prop_filtered` onto fine-scale intervals (e.g. use average within chunk)
    # FIXME: would it be better to have a fine-scale `prop_filtered`? Would need to smooth. 
    window_index = np.digitize(breakpoints[:-1], windows) - 1
    assert window_index.max() == windows.size - 2 and window_index.min() == 0
    prop_filtered = prop_filtered[window_index]
    # FIXME: need to adjust interval length to account for omitted variants, as these bases cannot carry
    # other mutations-- however doing so will mask these variants, which we want to retain in the
    # tree sequences. The bias should be minimal provided omitted variants are few.
else:
    breakpoints = windows.copy()
    logfile.write(
        f"{tag()} Adjusting per-chunk mutation rate to account for filtered sites "
        f"and masked intervals (for debugging only, will likely result in biased dates)\n"
    )
adj_mu = (1 - prop_inaccessible) * (1 - prop_filtered) * mutation_rate


# dump recombination rates, adjusted mutation rates, proportion inaccessible,
# proportion filtered, and chunk boundaries as RateMaps
adjusted_mu = msprime.RateMap(
    position=np.array(breakpoints),
    rate=np.array(adj_mu),
)
inaccessible = msprime.RateMap(
    position=np.array(breakpoints),
    rate=np.array(prop_inaccessible),
)
filtered = msprime.RateMap(
    position=np.array(breakpoints), 
    rate=np.array(prop_filtered),
)
chunks = msprime.RateMap(
    position=np.array(windows), 
    rate=filter_chunks.astype(float),
)
filter_windows = np.diff(adjusted_mu.get_cumulative_mass(statistics_windows)) > 0
stats_windows = msprime.RateMap(
    position=np.array(statistics_windows),
    rate=filter_windows.astype(float),
)
pickle.dump(hapmap, open(snakemake.output.recomb_rate, "wb"))
pickle.dump(adjusted_mu, open(snakemake.output.mut_rate, "wb"))
pickle.dump(inaccessible, open(snakemake.output.inaccessible, "wb"))
pickle.dump(filtered, open(snakemake.output.filtered, "wb"))
pickle.dump(chunks, open(snakemake.output.chunks, "wb"))
pickle.dump(stats_windows, open(snakemake.output.windows, "wb"))
pickle.dump(omitted_positions, open(snakemake.output.omitted, "wb"))

# FIXME: there is a bug in SINGER where fine-scale rate maps are never used
# and the mean rate is used instead. When this is fixed, explore using
# fine-scale rate maps (these may be problematic with zero rates). In the
# meantime, we explicitly use mean rates.

# dump SINGER parameters for each chunk
chunks_dir = snakemake.output.chunks_dir
os.makedirs(f"{chunks_dir}")
seeds = rng.integers(0, 2 ** 10, size=(filter_chunks.size, 2))
vcf_prefix = snakemake.output.vcf.removesuffix(".vcf")
for i in np.flatnonzero(filter_chunks):
    start, end = windows[i], windows[i + 1]
    polar = 0.99 if snakemake.params.polarised else 0.5
    id = f"{i:>06}"
    # SINGER truncates coordinates of SNPs, so adjust maps accordingly
    mut = adjusted_mu.slice(start, end, trim=True)
    rec = hapmap.slice(start, end, trim=True)
    chunk_path = os.path.join(chunks_dir, id)
    mutation_map_path = f"{chunk_path}.mut"
    recombination_map_path = f"{chunk_path}.rec"
    singer_params = {
        "thin": int(snakemake.params.singer_mcmc_thin), 
        "n": int(snakemake.params.singer_mcmc_samples),
        "Ne": float(Ne),
        # "m": str(0.0),    # NB: SINGER expects these arguments and toggles
        # "r": float(0.0),  # off map usage if at least one is positive
        # "mut_map": str(mutation_map_path),
        # "recomb_map": str(recombination_map_path),
        # see comment above
        "m": float(mut.mean_rate),
        "r": float(rec.mean_rate),
        "input": str(vcf_prefix), 
        "start": int(start), 
        "end": int(end), 
        "polar": float(polar),
        "seed": int(seeds[i, 0]),
        "output": str(chunk_path),
    }
    polegon_params = {
        "Ne": float(Ne),
        "mutation_map": str(mutation_map_path),
        "num_samples": int(snakemake.params.polegon_mcmc_samples),
        "burn_in": int(snakemake.params.polegon_mcmc_burnin),
        "thin": int(snakemake.params.polegon_mcmc_thin),
        "scaling_rep": int(snakemake.params.polegon_scaling_reps),
        "max_step": float(snakemake.params.polegon_max_step),
        "seed": int(seeds[i, 1]),
    }
    chunk_params = {"singer": singer_params, "polegon": polegon_params}
    yaml.dump(
        chunk_params, 
        open(f"{chunk_path}.yaml", "w"), 
        default_flow_style=False,
    )
    open(mutation_map_path, "w").write(ratemap_to_text(mut))
    open(recombination_map_path, "w").write(ratemap_to_text(rec))
    logfile.write(f"{tag()} Parameters for chunk {id} are {chunk_params}\n")


# if unpolarised, randomly flip reference and alternate
# TODO: ultimately take in an ancestral fasta and flip sites with ambiguous ancestral state
if not snakemake.params.polarised and snakemake.params.random_polarisation:
    flip_alleles = np.flatnonzero(rng.binomial(1, 0.5, size=retain.size))
    logfile.write(
        f"{tag()} Randomly choosing reference/alternate for all variants "
        f"to ensure half are mispolarised in input\n"
    )
    vcf['variants/REF'][flip_alleles], vcf['variants/ALT'][flip_alleles, 0] = \
        vcf['variants/ALT'][flip_alleles, 0], vcf['variants/REF'][flip_alleles]
    vcf['calldata/GT'][flip_alleles] = 1 - vcf['calldata/GT'][flip_alleles]


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


# dump reference and alternate allele per position
pickle.dump(
    {
        pos: (ref, alt) for pos, ref, alt in 
        zip(
            vcf['variants/POS'][retain], 
            vcf['variants/REF'][retain], 
            vcf['variants/ALT'][retain, 0],
        )
    },
    open(snakemake.output.alleles, "wb"),
)


# dump individual metadata
pickle.dump(metadata, open(snakemake.output.metadata, "wb"))


# dump statistics
diversity[~filter_windows] = np.nan
tajima_d[~filter_windows] = np.nan
vcf_stats = {
    "diversity": diversity, 
    "tajima_d": tajima_d, 
    "afs": afs, 
}
pickle.dump(vcf_stats, open(snakemake.output.vcf_stats, "wb"))


# calculate stratified statistics
# TODO: clean this up
vcf_strata_stats = {}
if stratify is not None:
    sample_sets = defaultdict(list)
    for i, md in enumerate(metadata):
        sample_sets[md[stratify]].append(i)
    strata = sorted([n for n in sample_sets.keys()])
    strata_counts = genotypes.count_alleles_subpops(sample_sets, max_allele=1)

    strata_divergence = []
    strata_afs = []
    for i in range(len(strata)):
        for j in range(i, len(strata)):
            divergence, *_ = allel.windowed_divergence(
                positions[statkeep],
                strata_counts[strata[i]][statkeep],
                strata_counts[strata[j]][statkeep],
                windows=statistics_windows_stack,
                is_accessible=statmask,
                fill=0.0,
            )
            divergence[~filter_windows] = np.nan
            strata_divergence.append(divergence)

        if snakemake.params.polarised:
            afs = allel.sfs(
                strata_counts[strata[i]][statkeep, 1], 
                n=2 * len(sample_sets[strata[i]]),
            ) / np.sum(statmask)
        else:
            afs = allel.sfs_folded(
                strata_counts[strata[i]][statkeep], 
                n=2 * len(sample_sets[strata[i]]),
            ) / np.sum(statmask)
        strata_afs.append(afs)

    strata_divergence = np.stack(strata_divergence).T
    vcf_strata_stats = {
        "strata": strata,
        "divergence": strata_divergence,
        "afs": strata_afs,
    }
pickle.dump(vcf_strata_stats, open(snakemake.output.vcf_strata_stats, "wb"))
