# Running the pipeline with SLURM

Snakemake supports using [SLURM](https://github.com/SchedMD/slurm) to
distribute jobs across HPC cluster nodes.  To run the pipeline on your SLURM
cluster you will need to:
1. Install Snakemake's SLURM executor plugin
2. Create a SLURM cluster profile
3. Tell Snakemake to use SLURM

*NOTE*: This assumes Snakemake version 8 or greater.

## Installing SLURM executor plugin

To install the SLURM executor plugin, Snakemake
[recommends](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)
you use `pip` within your `singer-snakemake-env` conda environment:

```bash
conda activate singer-snakemake-env
pip install snakemake-executor-plugin-slurm
```

## Creating a simple SLURM profile

You can either create a __global__ or __workflow__-specific profile in
Snakemake (see
[here](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles)
for more details).  Start with a simple profile that will set shared options
and resources for all steps in the pipeline, then adjust per rule as needed.

First, create a subdirectory within `singer-snakemake` called `simple_profile`
where your configuration file will be placed:

```bash
mkdir simple_profile
touch config.yaml
```

The profile configuration will depend on your HPC, the exact parameters in the
configuration file (e.g., chunk size), and the amount of data used in inference
(sample size, genome size, etc.).  For example, for 100 human genomes split
into 1Mb chunks, the following resources seem to be sufficient:

```yaml
# --- simple_profile/config.yaml ---
executor: slurm
jobs: 500  # maximum number of simultaneously running jobs.
max-jobs-per-second: 5  # maximum number of jobs to submit at once
max-status-checks-per-second: 0.5  # don't check too frequently
latency-wait: 10  # wait up to 10s for output files to appear
keep-going: true  # keep on going with jobs even if other jobs fail

# --- default resource settings ---
default-resources:
    slurm_account: <YOUR_ACCOUNT>
    slurm_partition: <PARTITION>
    cpus_per_task: 1
    mem_mb: 4000
    runtime: "6h" 
# NB: these are overridden by per-rule `resources:` blocks in the workflow.
```

## Run on cluster

At a minimum, the Snakemake invocation will look like:

```bash
snakemake --profile simple_profile --configfile config/example.yaml
```
