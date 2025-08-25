"""
Validation of the inference pipeline by simulating ARGs from stdpopsim,
inferring with SINGER, and comparing mutation age distributions.
Mutations frequencies are down-projected via hypergeometric sampling
to smooth the mutation age distributions.

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
MASK_SEQUENCE = config["mask-sequence"]
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
INFR_SITE_AFS_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.infr_site_afs.npy"
TRUE_SITE_AFS_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.true_site_afs.npy"
INFR_SITE_REL_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.infr_site_rel.npy"
TRUE_SITE_REL_PATH = f"{OUTPUT_DIR}/stats/{{chrom}}.true_site_rel.npy"
PLOT_PATH = f"{OUTPUT_DIR}/plots"


# --- rules --- #

rule all:
    input:
        vcf = expand(VCF_PATH, chrom=SIMULATION_SEEDS),
        trees = expand(INFR_TREES_PATH, chrom=SIMULATION_SEEDS, rep=MCMC_SAMPLES),
        pdf_plot = os.path.join(PLOT_PATH, "mutation-age-pdf.png"),
        exp_plot = os.path.join(PLOT_PATH, "mutation-age-expectation.png"),
        rel_plot = os.path.join(PLOT_PATH, "relatedness-over-time.png"),


rule simulate_arg:
    """
    Simulate multiple replicates from a stdpopsim demographic model with
    missing sequence/variants, write out the inputs to singer-snakemake
    pipeline.
    """
    output:
        vcf = VCF_PATH,
        trees = TRUE_TREES_PATH,
    params:
        mask_sequence = MASK_SEQUENCE,
        mask_variants = MASK_VARIANTS,
        config = STDPOPSIM_CONFIG,
    script:
        "validation/simulate_arg.py"


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
        trees = ancient(expand(INFR_TREES_PATH, rep=MCMC_SAMPLES, allow_missing=True)),
    output:
        site_afs = INFR_SITE_AFS_PATH,
    params:
        project_to = PROJECT_TO,
        time_grid = TIME_GRID,
        unknown_mutation_age = True,
        span_normalise = False,
    script:
        "validation/calculate_afs.py"


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
        span_normalise = False,
    script:
        "validation/calculate_afs.py"


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
        "validation/compare_mutation_ages.py"


rule calculate_inferred_relatedness:
    """
    Average time-windowed observed relatedness across MCMC samples per simulation.
    """
    input:
        trees = ancient(expand(INFR_TREES_PATH, rep=MCMC_SAMPLES, allow_missing=True)),
    output:
        site_relatedness = INFR_SITE_REL_PATH,
    params:
        time_grid = TIME_GRID,
        unknown_mutation_age = True,
        for_individuals = True,
        span_normalise = False,
    script:
        "validation/calculate_relatedness.py"


rule calculate_observed_relatedness:
    """
    Average time-windowed observed relatedness across MCMC samples per simulation.
    """
    input:
        trees = [TRUE_TREES_PATH],
    output:
        site_relatedness = TRUE_SITE_REL_PATH,
    params:
        time_grid = TIME_GRID,
        unknown_mutation_age = True,
        for_individuals = True,
        span_normalise = False,
    script:
        "validation/calculate_relatedness.py"


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
        "validation/compare_relatedness.py"
