"""
Simulate an ARG with demographic model, interval mask, and recombination rate
taken from custom input files. Note that the VCF/inferred ARG coordinate system
is one-based, but the true ARG coordinate system is zero-based.
"""

import stdpopsim
import msprime
import tszip
import demes
import gzip
import numpy as np
import warnings
import logging

from utils import simulate_sequence_mask
from utils import simulate_variant_mask
from utils import simulate_mispolarisation
from utils import ratemap_to_hapmap
from utils import population_metadata_csv
from utils import bitmask_to_bed
from utils import bed_to_bitmask
from utils import assert_valid_bedmask
from utils import assert_valid_hapmap
from utils import repolarise_tree_sequence
from utils import extract_ancestral_sequence

warnings.filterwarnings("ignore")
logging.basicConfig(
    filename=snakemake.log.logfile, 
    filemode="w", 
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]',
)

config = snakemake.params.config
model = msprime.Demography.from_demes(demes.load(config["demographic-model"]))
recombination_map = msprime.RateMap.read_hapmap(config["recombination-map"])
ploidy = config.get("ploidy", 2)

interval_mask_rate, interval_mask_length = snakemake.params.mask_sequence
record_structural_variants = snakemake.params.record_structural_variants
ancestral_mask_rate, ancestral_mask_length = snakemake.params.mask_ancestral
variant_mask_prop = snakemake.params.mask_variants
mispolarised_prop = snakemake.params.prop_mispolar
inaccessible_bed = snakemake.params.inaccessible_bed
unpolarised_bed = snakemake.params.unpolarised_bed
seed = int(snakemake.wildcards.chrom)
contig_name = snakemake.wildcards.chrom
logging.info(f"Log for instance {seed}")

# simulate data
samples = [
    msprime.SampleSet(n, population=p, ploidy=ploidy) 
    for p, n in config["samples"].items()
]
subseed = np.random.default_rng(seed).integers(2 ** 32 - 1, size=6)
prefix = snakemake.output.trees.removesuffix(".tsz")
ts = msprime.sim_ancestry(
    samples=samples,
    demography=model, 
    recombination_rate=recombination_map,
    random_seed=subseed[0],
)
ts = msprime.sim_mutations(
    ts,
    rate=config["mutation-rate"],
    random_seed=subseed[1],
)
logging.info(f"Simulated tree sequence:\n{ts}")

# add small structural variants and corresponding mask
ts, sv_mask, sequence_mask = simulate_sequence_mask(
    ts, 
    rate=interval_mask_rate, 
    length=interval_mask_length, 
    record_variants=record_structural_variants,
    seed=subseed[2], 
)
bed_to_bitmask(inaccessible_bed, sequence_mask) # applied on top of simulated mask
logging.info(f"Simulated sequence mask with {sequence_mask.sum()} missing bases")

# filter a random proportion of variants
variant_mask = simulate_variant_mask(ts, variant_mask_prop, subseed[3])

# adjust masks to remove overlap
site_position = ts.sites_position.astype(int)
assert sv_mask.size == variant_mask.size == site_position.size
sv_mask[sequence_mask[site_position]] = False
variant_mask[sequence_mask[site_position]] = False
variant_mask[sv_mask] = False
logging.info(f"Simulated variant mask with {variant_mask.sum()} missing sites")

# filter out masked sites from true trees, for the sake of downstream comparison
site_masked = np.logical_or(sequence_mask[site_position], variant_mask)
ts = ts.delete_sites(np.flatnonzero(site_masked))
logging.info(f"Tree sequence after filtering sites:\n{ts}")

# simulate ancestral sequence
ancestral_sequence = extract_ancestral_sequence(ts)
_, _, ancestral_mask = simulate_sequence_mask(
    ts,
    rate=ancestral_mask_rate,
    length=ancestral_mask_length,
    seed=subseed[4],
)
bed_to_bitmask(unpolarised_bed, ancestral_mask) # applied on top of simulated mask
ancestral_sequence[ancestral_mask] = "N"
logging.info(f"Simulated ancestral mask with {ancestral_mask.sum()} missing states")

# simulate sites to repolarise
repolarise = simulate_mispolarisation(ts, mispolarised_prop, subseed[5])
logging.info(f"Simulated repolarisation of {repolarise.sum()} sites")

# write out sequence mask as bed
bedmask = open(f"{prefix}.mask.bed", "w")
bedmask.write(bitmask_to_bed(sequence_mask, contig_name))
bedmask.close()
assert_valid_bedmask(sequence_mask, f"{prefix}.mask.bed")
logging.info(f"Wrote sequence mask to {prefix}.mask.bed")

# write out variants to omit from dating as one-based positions
omitted = open(f"{prefix}.omit.txt", "w")
omitted.write("\n".join([str(x + 1) for x in site_position[sv_mask]]) + "\n")
omitted.close()
logging.info(f"Wrote list of omitted sites to {prefix}.omit.txt")

# write out variant mask as one-based positions
sitemask = open(f"{prefix}.filter.txt", "w")
sitemask.write("\n".join([str(x + 1) for x in site_position[variant_mask]]) + "\n")
sitemask.close()
logging.info(f"Wrote list of filtered sites to {prefix}.filter.txt")

# write out hapmap
hapmap = open(f"{prefix}.hapmap", "w")
hapmap.write(ratemap_to_hapmap(recombination_map, contig_name, missing_as_zero=True))
hapmap.close()
assert_valid_hapmap(recombination_map, f"{prefix}.hapmap", ignore_missing=True)
logging.info(f"Wrote hapmap to {prefix}.hapmap")

# write out metadata as csv
metadata_csv, individual_names = population_metadata_csv(ts)
metadata = open(f"{prefix}.meta.csv", "w")
metadata.write(metadata_csv)
metadata.close()
logging.info(f"Wrote metadata to {prefix}.meta.csv")

# write out ancestral sequence
if not snakemake.params.skip_ancestral_sequence:
    ancestral = gzip.open(f"{prefix}.ancestral.fa.gz", "wt")
    ancestral.write(f">{contig_name}\n" + "".join(ancestral_sequence))
    ancestral.close()
    logging.info(f"Wrote ancestral states to {prefix}.ancestral.fa.gz")

# write out trees
tszip.compress(ts, snakemake.output.trees)
logging.info(f"Wrote tree sequence to {snakemake.output.trees}")

# mispolarise and write out VCF
repolarise_tree_sequence(ts, repolarise).write_vcf(
    gzip.open(f"{prefix}.vcf.gz", "wt"), 
    contig_id=contig_name,
    individual_names=individual_names,
    position_transform=lambda x: 1 + np.round(x).astype(np.int64),  # 1-based positions
)
logging.info(f"Wrote repolarised vcf to {prefix}.vcf.gz")
