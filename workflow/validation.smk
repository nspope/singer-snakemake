"""
Validation of the inference pipeline by simulating ARGs from stdpopsim,
inferring with SINGER, and comparing distributions of mutation ages and
repolarised ancestral states across frequencies.

Mutations frequencies are down-projected via hypergeometric sampling
to smooth the mutation age distributions.

Pair coalescence time distributions are also compared.

Part of https://github.com/nspope/singer-snakemake.
"""

import os
import yaml
import stdpopsim
import numpy as np

from msprime import RateMap
from snakemake.utils import min_version
min_version("8.0")


# --- parse config --- #

# configfile: "config/validation_stdpopsim.yaml"

OUTPUT_DIR = config["output-dir"]
INPUT_DIR = os.path.join(OUTPUT_DIR, "simulated-data")
RANDOM_SEED = int(config["random-seed"])
NUM_REPLICATES = int(config["num-replicates"])
MASK_SEQUENCE = config["mask-sequence"]
MASK_VARIANTS = float(config["mask-variants"])
INACCESSIBLE_BED = config.get("inaccessible-bed", None)
PROP_MISPOLAR = config["prop-mispolarise"]
TIME_GRID = np.append(np.logspace(*config["time-grid"]), np.inf)
PROJECT_TO = config["project-afs-to"]

# set up simulation context
if "stdpopsim-config" in config:
    SIMULATION_CONFIG = config["stdpopsim-config"]
    REF_SPECIES = stdpopsim.get_species(SIMULATION_CONFIG["species"])
    REF_CONTIG = REF_SPECIES.get_contig(**SIMULATION_CONFIG["contig"])
    MUTATION_RATE = REF_CONTIG.mutation_rate
    RECOMBINATION_RATE = REF_CONTIG.recombination_map.mean_rate
    SIMULATION_SCRIPT = "scripts/validation/simulate_stdpopsim.py"
elif "msprime-config" in config:
    SIMULATION_CONFIG = config["msprime-config"]
    MUTATION_RATE = SIMULATION_CONFIG["mutation-rate"]
    RECOMBINATION_MAP = SIMULATION_CONFIG["recombination-map"]
    RECOMBINATION_RATE = RateMap.read_hapmap(RECOMBINATION_MAP).mean_rate
    SIMULATION_SCRIPT = "scripts/validation/simulate_msprime.py"
else:
    raise ValueError("No simulation config provided")

# set up simulation design
RNG = np.random.default_rng(RANDOM_SEED)
SIMULATION_SEEDS = RNG.integers(2 ** 32 - 1, size=NUM_REPLICATES)
SINGER_SNAKEMAKE_SEED = RNG.integers(2 ** 32 - 1, size=1).item()

# set singer-snakemake parameters
SINGER_CONFIG_OVERRIDE = config.get("singer-snakemake-config-override", None)
SINGER_CONFIG = yaml.safe_load(open(config["singer-snakemake-config"]))
if SINGER_CONFIG_OVERRIDE is not None: 
    SINGER_CONFIG.update(SINGER_CONFIG_OVERRIDE)
SINGER_CONFIG["input-dir"] = INPUT_DIR
SINGER_CONFIG["output-dir"] = OUTPUT_DIR
SINGER_CONFIG["mutation-rate"] = MUTATION_RATE
SINGER_CONFIG["recombination-rate"] = RECOMBINATION_RATE
SINGER_CONFIG["random-seed"] = SINGER_SNAKEMAKE_SEED
SINGER_CONFIG["chromosomes"] = [str(x) for x in SIMULATION_SEEDS]
BURN_IN = int(SINGER_CONFIG["singer-mcmc-burnin"] * SINGER_CONFIG["singer-mcmc-samples"])
MCMC_SAMPLES = np.arange(SINGER_CONFIG["singer-mcmc-samples"])[BURN_IN:]

# enumerate intermediates
VCF_PATH = f"{INPUT_DIR}/{{chrom}}.vcf.gz"
TRUE_TREES_PATH = f"{INPUT_DIR}/{{chrom}}.tsz"
INFR_TREES_PATH = f"{OUTPUT_DIR}/{{chrom}}/trees/{{chrom}}.{{rep}}.tsz"
INACCESSIBLE_PATH = f"{OUTPUT_DIR}/{{chrom}}/{{chrom}}.inaccessible.p"
VCF_ALLELES_PATH = f"{OUTPUT_DIR}/{{chrom}}/{{chrom}}.alleles.p"
DIAGNOSTICS_PATH = f"{OUTPUT_DIR}/{{chrom}}/plots/repolarised-trace.png"
INFR_SITE_AFS_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.infr_site_afs.npy"
TRUE_SITE_AFS_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.true_site_afs.npy"
INFR_SITE_REL_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.infr_site_rel.npy"
TRUE_SITE_REL_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.true_site_rel.npy"
INFR_PAIR_DEN_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.infr_pair_den.npy"
TRUE_PAIR_DEN_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.true_pair_den.npy"
INFR_PAIR_RAT_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.infr_pair_rat.npy"
TRUE_PAIR_RAT_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.true_pair_rat.npy"
MISPOLARISED_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.mispolarised.npy"
PLOT_PATH = f"{OUTPUT_DIR}/plots"


# --- rules --- #

rule all:
    input:
        vcf = expand(VCF_PATH, chrom=SIMULATION_SEEDS),
        trees = ancient(expand(INFR_TREES_PATH, chrom=SIMULATION_SEEDS, rep=MCMC_SAMPLES)),
        inaccessible = ancient(expand(INACCESSIBLE_PATH, chrom=SIMULATION_SEEDS)),
        diagnostics = ancient(expand(DIAGNOSTICS_PATH, chrom=SIMULATION_SEEDS)),
        alleles = ancient(expand(VCF_ALLELES_PATH, chrom=SIMULATION_SEEDS)),
        pdf_plot = os.path.join(PLOT_PATH, "mutation-age-pdf.png"),
        exp_plot = os.path.join(PLOT_PATH, "mutation-age-expectation.png"),
        rel_plot = os.path.join(PLOT_PATH, "relatedness-over-time.png"),
        pol_plot = os.path.join(PLOT_PATH, "proportion-mispolarised.png"),
        pair_den_plot = os.path.join(PLOT_PATH, "proportion-coalescing-pairs.png"),
        pair_rat_plot = os.path.join(PLOT_PATH, "pair-coalescence-rates.png"),


rule simulate_arg:
    """
    Simulate multiple replicates from a demographic model with missing
    sequence/variants, write out the inputs to singer-snakemake pipeline.
    """
    output:
        vcf = VCF_PATH,
        trees = TRUE_TREES_PATH,
    params:
        mask_sequence = MASK_SEQUENCE,
        mask_variants = MASK_VARIANTS,
        prop_mispolar = PROP_MISPOLAR,
        inaccessible_bed = INACCESSIBLE_BED,
        config = SIMULATION_CONFIG,
    script: SIMULATION_SCRIPT


module singer_snakemake:
    """
    Use SINGER to infer ARGs for each simulation.
    """
    snakefile: "Snakefile"
    config: SINGER_CONFIG

use rule * from singer_snakemake exclude all


rule calculate_inferred_afs:
    """
    Average time-windowed observed SFS across MCMC samples per simulation,
    after projecting down to smaller sample sizes per population.
    """
    input:
        trees = expand(INFR_TREES_PATH, rep=MCMC_SAMPLES, allow_missing=True),
    output:
        site_afs = INFR_SITE_AFS_PATH,
    params:
        project_to = PROJECT_TO,
        time_grid = TIME_GRID,
        unknown_mutation_age = True,
    script:
        "scripts/validation/calculate_afs.py"


rule calculate_reference_afs:
    """
    Caclulate time-windowed observed SFS from true ARG, after projecting down to
    smaller sample sizes per population.
    """
    input:
        trees = [TRUE_TREES_PATH],
    output:
        site_afs = TRUE_SITE_AFS_PATH,
    params:
        project_to = PROJECT_TO,
        time_grid = TIME_GRID,
        unknown_mutation_age = False,
    script:
        "scripts/validation/calculate_afs.py"


rule compare_mutation_ages:
    """
    Plot mutation age distribution for each entry in the down-projected AFS.
    """
    input:
        true_site_afs = expand(TRUE_SITE_AFS_PATH, chrom=SIMULATION_SEEDS),
        infr_site_afs = expand(INFR_SITE_AFS_PATH, chrom=SIMULATION_SEEDS),
    output:
        pdf_plot = rules.all.input.pdf_plot,
        exp_plot = rules.all.input.exp_plot,
    params:
        time_grid = TIME_GRID,
    script:
        "scripts/validation/compare_mutation_ages.py"


rule calculate_inferred_relatedness:
    """
    Average time-windowed observed relatedness across MCMC samples per simulation.
    """
    input:
        trees = expand(INFR_TREES_PATH, rep=MCMC_SAMPLES, allow_missing=True),
    output:
        site_relatedness = INFR_SITE_REL_PATH,
    params:
        time_grid = TIME_GRID,
        unknown_mutation_age = True,
        for_individuals = True,
    script:
        "scripts/validation/calculate_relatedness.py"


rule calculate_reference_relatedness:
    """
    Calculate time-windowed observed relatedness for true ARGs.
    """
    input:
        trees = [TRUE_TREES_PATH],
    output:
        site_relatedness = TRUE_SITE_REL_PATH,
    params:
        time_grid = TIME_GRID,
        unknown_mutation_age = True,
        for_individuals = True,
    script:
        "scripts/validation/calculate_relatedness.py"


rule compare_relatedness:
    """
    Plot individual relatedness across time for true and inferred ARGs.
    """
    input:
        true_site_relatedness = expand(TRUE_SITE_REL_PATH, chrom=SIMULATION_SEEDS),
        infr_site_relatedness = expand(INFR_SITE_REL_PATH, chrom=SIMULATION_SEEDS),
    output:
        rel_plot = rules.all.input.rel_plot,
    params:
        time_grid = TIME_GRID,
        log_relatedness = False,
        max_individuals = 6,
    script:
        "scripts/validation/compare_relatedness.py"


rule calculate_mispolarised:
    """
    Average the number of variants correctly or incorrectly repolarised
    conditional on true frequency, across MCMC iterations.
    """
    input:
        vcf_alleles = VCF_ALLELES_PATH,
        infr_trees = expand(INFR_TREES_PATH, rep=MCMC_SAMPLES, allow_missing=True),
        true_trees = TRUE_TREES_PATH,
    output:
        mispolarised = MISPOLARISED_PATH,
    params:
        project_to = PROJECT_TO,
        position_adjust = 1,  # position offset applied to true trees
    script:
        "scripts/validation/calculate_mispolarised.py"


rule compare_mispolarised:
    """
    Plot the proportions of mispolarised variants across frequencies.
    """
    input:
        mispolarised = expand(MISPOLARISED_PATH, chrom=SIMULATION_SEEDS),
    output:
        pol_plot = rules.all.input.pol_plot,
    script:
        "scripts/validation/compare_mispolarised.py"


rule calculate_inferred_pair_coalescence:
    """
    Average pair coalescence time distribution across MCMC replicates.
    """
    input:
        trees = expand(INFR_TREES_PATH, rep=MCMC_SAMPLES, allow_missing=True),
        inaccessible = INACCESSIBLE_PATH,
    output:
        pair_density = INFR_PAIR_DEN_PATH,
        pair_rates = INFR_PAIR_RAT_PATH,
    params:
        time_grid = TIME_GRID,
    script:
        "scripts/validation/calculate_pair_coalescence.py"


rule calculate_reference_pair_coalescence:
    """
    Calculate pair coalescence time distribution for true ARG.
    """
    input:
        trees = [TRUE_TREES_PATH],
        inaccessible = INACCESSIBLE_PATH,
    output:
        pair_density = TRUE_PAIR_DEN_PATH,
        pair_rates = TRUE_PAIR_RAT_PATH,
    params:
        time_grid = TIME_GRID,
    script:
        "scripts/validation/calculate_pair_coalescence.py"


rule compare_pair_coalescence:
    """
    Plot true and inferred coalescence time distributions.
    """
    input:
        infr_pair_density = expand(INFR_PAIR_DEN_PATH, chrom=SIMULATION_SEEDS),
        true_pair_density = expand(TRUE_PAIR_DEN_PATH, chrom=SIMULATION_SEEDS),
        infr_pair_rates = expand(INFR_PAIR_RAT_PATH, chrom=SIMULATION_SEEDS),
        true_pair_rates = expand(TRUE_PAIR_RAT_PATH, chrom=SIMULATION_SEEDS),
    output:
        pair_den_plot = rules.all.input.pair_den_plot,
        pair_rat_plot = rules.all.input.pair_rat_plot,
    params:
        time_grid = TIME_GRID,
        log_coalescence_rates = True,
    script:
        "scripts/validation/compare_pair_coalescence.py"
