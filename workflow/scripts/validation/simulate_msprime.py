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

warnings.filterwarnings("ignore")

config = snakemake.params.config
model = msprime.Demography.from_demes(demes.load(config["demographic-model"]))
recombination_map = msprime.RateMap.read_hapmap(config["recombination-map"])

interval_mask_density, interval_mask_length = snakemake.params.mask_sequence
variant_mask_prop = snakemake.params.mask_variants
mispolarised_prop = snakemake.params.prop_mispolar
inaccessible_bed = snakemake.params.inaccessible_bed
seed = int(snakemake.wildcards.chrom)
contig_name = snakemake.wildcards.chrom

# simulate data
subseed = np.random.default_rng(seed).integers(2 ** 32 - 1, size=5)
prefix = snakemake.output.trees.removesuffix(".tsz")
ts = msprime.sim_ancestry(
    samples=config["samples"],
    demography=model, 
    recombination_rate=recombination_map,
    random_seed=subseed[0],
)
ts = msprime.sim_mutations(
    ts,
    rate=config["mutation-rate"],
    random_seed=subseed[4],
)

sequence_mask = simulate_sequence_mask(ts, interval_mask_density, interval_mask_length, subseed[1])
bed_to_bitmask(inaccessible_bed, sequence_mask) # applied on top of simulated mask
variant_mask = simulate_variant_mask(ts, variant_mask_prop, subseed[2])

# filter out masked sites from true trees, for the sake of downstream comparison
site_position = ts.sites_position.astype(int)
site_masked = np.logical_or(sequence_mask[site_position], variant_mask)
ts = ts.delete_sites(np.flatnonzero(site_masked))

# write out sequence mask as bed
bedmask = open(f"{prefix}.mask.bed", "w")
bedmask.write(bitmask_to_bed(sequence_mask, contig_name))
bedmask.close()
assert_valid_bedmask(sequence_mask, f"{prefix}.mask.bed")

# write out variant mask as one-based positions
sitemask = open(f"{prefix}.filter.txt", "w")
sitemask.write("\n".join([str(x + 1) for x in site_position[variant_mask]]) + "\n")
sitemask.close()

# write out hapmap
hapmap = open(f"{prefix}.hapmap", "w")
hapmap.write(ratemap_to_hapmap(recombination_map, contig_name, missing_as_zero=True))
hapmap.close()
assert_valid_hapmap(recombination_map, f"{prefix}.hapmap", ignore_missing=True)

# write out metadata as csv
metadata_csv, individual_names = population_metadata_csv(ts)
metadata = open(f"{prefix}.meta.csv", "w")
metadata.write(metadata_csv)
metadata.close()

# write out trees
tszip.compress(ts, snakemake.output.trees)

# mispolarise and write out VCF
repolarise = simulate_mispolarisation(ts, mispolarised_prop, subseed[3])
repolarise_tree_sequence(ts, repolarise).write_vcf(
    gzip.open(f"{prefix}.vcf.gz", "wt"), 
    contig_id=contig_name,
    individual_names=individual_names,
    position_transform=lambda x: 1 + np.round(x).astype(np.int64),  # 1-based positions
)
