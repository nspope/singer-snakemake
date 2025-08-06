"""
Validation of the inference pipeline (in particular the scheme for accounting for missing data)
by simulating ARGs from stdpopsim, inferring with SINGER, and comparing mutation age distributions.
Mutations frequencies are down-projected via hypergeometric sampling to smooth the mutation age
distributions.

Part of https://github.com/nspope/singer-snakemake.
"""

import os
import yaml
import stdpopsim
import numpy as np

from snakemake.utils import min_version
min_version("8.0")


# --- parse config --- #

configfile: "config/validation.yaml"

OUTPUT_DIR = config["output-dir"]
INPUT_DIR = os.path.join(OUTPUT_DIR, "simulated-data")
RANDOM_SEED = int(config["random-seed"])
NUM_REPLICATES = int(config["num-replicates"])
SEQUENCE_LENGTH = float(config["sequence-length"])
MASK_SEQUENCE = float(config["mask-sequence"])
MASK_VARIANTS = float(config["mask-variants"])
TIME_GRID = np.append(np.logspace(*config["time-grid"]), np.inf)
PROJECT_TO = int(config["project-afs-to"])

# set up simulation context
STDPOPSIM_CONFIG = config["stdpopsim-config"]
REF_SPECIES = stdpopsim.get_species(STDPOPSIM_CONFIG["species"])
REF_CONTIG = REF_SPECIES.get_contig(**STDPOPSIM_CONFIG["contig"])

# set up simulation design
RNG = np.random.default_rng(RANDOM_SEED)
SIMULATION_SEEDS = RNG.integers(2 ** 32 - 1, size=NUM_REPLICATES)
SINGER_SNAKEMAKE_SEED = RNG.integers(2 ** 32 - 1, size=1).item()

# set singer-snakemake parameters
SINGER_CONFIG = yaml.safe_load(open(config["singer-snakemake-config"]))
SINGER_CONFIG["input-dir"] = INPUT_DIR
SINGER_CONFIG["output-dir"] = OUTPUT_DIR
SINGER_CONFIG["mutation-rate"] = REF_CONTIG.mutation_rate
SINGER_CONFIG["recombination-rate"] = REF_CONTIG.recombination_map.mean_rate
SINGER_CONFIG["random-seed"] = SINGER_SNAKEMAKE_SEED
SINGER_CONFIG["chromosomes"] = [str(x) for x in SIMULATION_SEEDS]
BURN_IN = int(SINGER_CONFIG["mcmc-burnin"] * SINGER_CONFIG["mcmc-samples"])
MCMC_SAMPLES = np.arange(SINGER_CONFIG["mcmc-samples"])[BURN_IN:]

# enumerate intermediates
VCF_PATH = f"{INPUT_DIR}/{{chrom}}.vcf.gz"
TRUE_TREES_PATH = f"{INPUT_DIR}/{{chrom}}.tsz"
INFR_TREES_PATH = f"{OUTPUT_DIR}/{{chrom}}/trees/{{chrom}}.{{rep}}.tsz"
INFR_BRANCH_AFS_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.infr_branch_afs.npy"
TRUE_BRANCH_AFS_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.true_branch_afs.npy"
INFR_SITE_AFS_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.infr_site_afs.npy"
TRUE_SITE_AFS_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.true_site_afs.npy"
PLOT_PATH = f"{OUTPUT_DIR}/plots"


# --- rules --- #

rule all:
    input:
        vcf = expand(VCF_PATH, chrom=SIMULATION_SEEDS),
        trees = expand(INFR_TREES_PATH, chrom=SIMULATION_SEEDS, rep=MCMC_SAMPLES),
        age_plot = os.path.join(PLOT_PATH, "mutation-ages.png"),


rule simulate_arg:
    """
    Simulate multiple replicates from a stdpopsim demographic model with
    missing sequence/variants, write out the inputs to singer-snakemake
    pipeline.
    """
    output:
        vcf = expand(VCF_PATH, chrom=SIMULATION_SEEDS),
        trees = expand(TRUE_TREES_PATH, chrom=SIMULATION_SEEDS),
    params:
        mask_sequence = MASK_SEQUENCE,
        mask_variants = MASK_VARIANTS,
        seeds = SIMULATION_SEEDS,
        config = STDPOPSIM_CONFIG,
    script:
        "validation/simulate_arg.py"


module singer_snakemake:
    """
    Use SINGER to infer ARG.
    """
    TODO
    snakefile: 
        github("nspope/singer-snakemake", path="workflow/Snakefile", commit="57ad022")
    config: SINGER_CONFIG

use rule * from singer_snakemake exclude all


rule calculate_afs:
    """
    Average time windowed SFS across MCMC samples per simulation, after
    projecting down to smaller sample sizes.
    """
    input:
        truth = TRUE_TREES_PATH,
        trees = expand(INFR_TREES_PATH, rep=MCMC_SAMPLES, allow_missing=True),
    output:
        infr_branch_afs = INFR_BRANCH_AFS_PATH,
        true_branch_afs = TRUE_BRANCH_AFS_PATH,
        infr_site_afs = INFR_SITE_AFS_PATH,
        true_site_afs = TRUE_SITE_AFS_PATH,
    params:
        project_to = PROJECT_TO,
        time_grid = TIME_GRID,
        mutation_rate = MUTATION_RATE,
    script:
        "validation/calculate_afs.py"


rule compare_mutation_ages:
    """
    Plot mutation age distribution for each entry in the down-projected AFS,
    as a sanity check.
    """
    input:
        true_branch_afs = expand(TRUE_BRANCH_AFS_PATH, chrom=SIMULATION_SEEDS),
        infr_branch_afs = expand(INFR_BRANCH_AFS_PATH, chrom=SIMULATION_SEEDS),
        true_site_afs = expand(TRUE_SITE_AFS_PATH, chrom=SIMULATION_SEEDS),
        infr_site_afs = expand(INFR_SITE_AFS_PATH, chrom=SIMULATION_SEEDS),
    output:
        age_plot = rules.all.input.age_plot,
    params:
        time_grid = TIME_GRID,
    script:
        "validation/compare_mutation_ages.py"
