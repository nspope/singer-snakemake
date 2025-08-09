"""
Simulate an ARG with missing data/sequence. Note that the VCF/inferred ARG
coordinate system is one-based, but the true ARG coordinate system is
zero-based.
"""

import stdpopsim
import tszip
import gzip
import numpy as np
import warnings

from utils import simulate_sequence_mask
from utils import simulate_variant_mask
from utils import ratemap_to_hapmap
from utils import population_metadata_csv
from utils import bitmask_to_bed
from utils import assert_valid_bedmask
from utils import assert_valid_hapmap

warnings.filterwarnings("ignore")


config = snakemake.params.config
species = stdpopsim.get_species(config["species"])
demography = species.get_demographic_model(config["demographic-model"])
contig = species.get_contig(**config["contig"])
engine = stdpopsim.get_engine("msprime")  #FIXME: use SLiM if DFE
contig_name, *_ = contig.coordinates

interval_mask_density, interval_mask_length = snakemake.params.mask_sequence
variant_mask_prop = snakemake.params.mask_variants
seed = int(snakemake.wildcards.chrom)

# simulate data
subseed = np.random.default_rng(seed).integers(2 ** 32 - 1, size=3)
prefix = snakemake.output.trees.removesuffix(".tsz")
ts = engine.simulate(
    demographic_model=demography, 
    contig=contig, 
    samples=config["samples"],
    seed=subseed[0],
)
sequence_mask = simulate_sequence_mask(ts, interval_mask_density, interval_mask_length, subseed[1])
variant_mask = simulate_variant_mask(ts, variant_mask_prop, subseed[2])
print("foo", variant_mask_prop, variant_mask.sum())

# filter out masked sites from true trees, for the sake of downstream comparison
site_position = ts.sites_position.astype(int)
site_masked = np.logical_and(
    sequence_mask[site_position],
    variant_mask,
)
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
hapmap.write(ratemap_to_hapmap(contig.recombination_map, contig_name, missing_as_zero=True))
hapmap.close()
assert_valid_hapmap(contig.recombination_map, f"{prefix}.hapmap", ignore_missing=True)

# write out metadata as csv
metadata_csv, individual_names = population_metadata_csv(ts)
metadata = open(f"{prefix}.meta.csv", "w")
metadata.write(metadata_csv)
metadata.close()

# write out vcf and trees
ts.write_vcf(
    gzip.open(f"{prefix}.vcf.gz", "wt"), 
    contig_id=contig_name,
    individual_names=individual_names,
    position_transform=lambda x: 1 + np.round(x).astype(np.int64),  # 1-based positions
)
tszip.compress(ts, snakemake.output.trees)
