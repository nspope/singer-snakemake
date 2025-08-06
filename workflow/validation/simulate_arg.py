"""
Simulate an ARG with missing data/sequence.
"""

import stdpopsim
import tszip
import gzip
import numpy as np
import warnings

from utils import simulate_sequence_mask
from utils import simulate_variant_mask
from utils import ratemap_to_hapmap
from utils import population_metadata

config = snakemake.params.config

species = stdpopsim.get_species(config["species"])
demography = species.get_demographic_model(config["demographic-model"])
contig = species.get_contig(**config["contig"])
engine = stdpopsim.get_engine("msprime")  #FIXME: use SLiM if DFE


for seed, trees in zip(snakemake.params.seeds, snakemake.output.trees):

    prefix = snakemake.output.trees.removesuffix(".tsz")

    ts = engine.simulate(
        demographic_model=demography, 
        contig=contig, 
        samples=config["samples"],
        seed=seed,
    )

    # generate sequence mask
    if snakemake.params.mask_sequence > 0:
        assert "inaccessible_mask" not in config["contig"], "Sequence mask already exists"
        assert "accessible_mask" not in config["contig"], "Sequence mask already exists"
        sequence_mask = simulate_sequence_mask(...)

    # generate variant mask
    if snakemake.params.mask_variants > 0:
        variant_mask = simulate_variant_mask(...)

    # write out hapmap
    hapmap.open(f"{prefix}.hapmap", "w")
    hapmap.write(ratemap_to_hapmap(contig.recombination_map))
    hapmap.close()

    # write out metadata
    metadata_str, individual_names = population_metadata(ts)
    metadata.open(f"{prefix}.meta.csv", "w")
    metadata.write(metadata_str)
    metadata.close()

    # write out vcf and trees
    ts.write_vcf(gzip.open(f"{prefix}.vcf.gz", "wt"), individual_names=individual_names)
    tszip.compress(ts, trees)



