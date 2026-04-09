# Running the pipeline with SLURM

Snakemake supports using [SLURM](https://github.com/SchedMD/slurm) to distribute jobs across HPC cluster nodes.
To run the pipeline on your SLURM cluster you will need:
1. Install Snakemake's SLURM executor plugin
2. Create a SLURM cluster profile
3. Tell Snakemake to use SLURM

*Note*: This assumes Snakemake 8+

## Installing SLURM executor plugin

To install the SLURM executor plugin, Snakemake [recommends](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) you use `pip` within your `singer-snakemake-env` conda environment:

```bash
conda activate singer-snakemake-env
pip install snakemake-executor-plugin-slurm
```

## Creating a simple SLURM profile

You can either create a __global__ or __workflow__-specific profile in Snakemake (see [here](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles) for more details).
Start with a simple profile that will set shared options and resources for all steps in the pipeline, then adjust per rule as needed.

First, create a subdirectory within `singer-snakemake` called `simple_profile` where your configuration file will be placed:

```bash
mkdir simple_profile
touch config.yaml
```

The profile configuration will depend on your HPC, the exact parameters in the configuration file (e.g., chunk size), 
and the amount of data used in inference (sample size, genome size, etc.). 
For example, for 100 human genomes split into 1Mb chunks, the following resources seem to be sufficient (to be added to your `config.yaml`):

```yaml
# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
executor: slurm

# ---------------------------------------------------------------------------
# Job control
# ---------------------------------------------------------------------------
# Maximum number of simultaneously running jobs.
jobs: 500

# Maximum number of jobs to submit at once (Slurm queues them, but submitting
# thousands at once is wasteful and slows down the scheduler).
max-jobs-per-second: 5
max-status-checks-per-second: 0.5

# ---------------------------------------------------------------------------
# Fault tolerance
# ---------------------------------------------------------------------------
# Wait up to 10 s for output files to appear on the shared filesystem after
# a job finishes (takes possible metadata latency into account).
latency-wait: 10

# Keep going with independent jobs even if some jobs fail.
keep-going: True

# ---------------------------------------------------------------------------
# Default Slurm resource settings
# These are overridden by per-rule `resources:` blocks in the workflow.
# ---------------------------------------------------------------------------

default-resources:
    slurm_account: <YOUR_ACCOUNT>
    slurm_partition: <PARTITION>
    cpus_per_task: 1
    mem_mb: 4_000 # allocated RAM in Mb
    runtime: "6h" # Run

```

## Ready to run!

Now you are ready to run the pipeline on your SLURM cluster.
At minimum, your command-line Snakemake call will look something like:

```bash
snakemake --profile simple_profile --configfile config/example.yaml
```