"""
Run POLEGON on a chunk of ARG produced by SINGER.

Part of https://github.com/nspope/singer-snakemake.
"""

import os
import shutil
import yaml
import subprocess
import numpy as np
import msprime
import tszip
import tskit
from datetime import datetime

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

logfile = open(snakemake.log.log, "w")
use_polegon = snakemake.params.use_polegon
use_mutational_span = snakemake.params.use_mutational_span

if use_polegon:
    params = yaml.safe_load(open(snakemake.input.params)).pop("polegon")
    seed = params.pop("seed") + int(snakemake.wildcards.rep)
    # FIXME: Ne is lower by factor of two relative to `polegon_master`.
    # This shouldn't matter, it cancels during rescaling, but look into it.

    # POLEGON expects inputs named slightly differently than SINGER output
    prefix = snakemake.input.muts.replace("_muts_", "_").removesuffix(".txt")
    shutil.copy(snakemake.input.muts, f"{prefix}_muts.txt")
    shutil.copy(snakemake.input.nodes, f"{prefix}_nodes.txt")

    # Adjust branch spans to reflect masked sequence, and absorb mutation rate
    # into span.  This is necessary because POLEGON doesn't use the "average
    # branch mutation rate" from the mutation map during rescaling, and instead
    # just adjusts by the mean rate. Manually converting the branch spans to
    # mutational units fixes this.
    if use_mutational_span:
        logfile.write(
            f"{tag()} Adjusting branch spans input ({prefix}_branches.txt) "
            f"to reflect mutation rate map and setting mutation rate to unity\n"
        )
        params["m"] = 1.0
        mutation_map = params.pop("mutation_map")
        adjusted_mu = np.loadtxt(mutation_map, ndmin=2)
        assert np.all(adjusted_mu[:-1, 1] == adjusted_mu[1:, 0])
        adjusted_mu = msprime.RateMap(
           position=np.append(adjusted_mu[0, 0], adjusted_mu[:, 1]),
           rate=adjusted_mu[:, 2],
        )
        branches = np.loadtxt(snakemake.input.branches)
        for i in range(2):
           branches[:, i] = adjusted_mu.get_cumulative_mass(branches[:, i])
        np.savetxt(f"{prefix}_branches.txt", branches)
    else:
        logfile.write(
            f"{tag()} Using mutation rate map without adjusting branch spans input "
            f"(for debugging only, will likely produce biased node ages)\n"
        )
        shutil.copy(snakemake.input.branches, f"{prefix}_branches.txt")

    invocation = [
        f"{snakemake.params.polegon_binary}",
        "-input", f"{prefix}",
        "-seed", f"{seed}",
    ]
    for arg, val in params.items():
        invocation += f"-{arg} {val}".split()
    
    logfile.write(f"{tag()} " + " ".join(invocation) + "\n")
    process = subprocess.run(invocation, check=False, stdout=logfile, stderr=logfile)
    logfile.write(f"{tag()} POLEGON run ended ({process.returncode})\n")

    for suffix in ["muts", "branches", "nodes"]:
        os.remove(f"{prefix}_{suffix}.txt")

    assert process.returncode == 0, f"POLEGON terminated with error ({process.returncode})"
    os.rename(f"{prefix}_new_nodes.txt", snakemake.output.nodes)
else:
    logfile.write(f"{tag()} Skipping dating with POLEGON\n")
    shutil.copy(snakemake.input.nodes, snakemake.output.nodes)
