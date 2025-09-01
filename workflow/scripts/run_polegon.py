"""
Run POLEGON on a SINGER sample and compute a 'expected' ARG.
"""

import os
import pickle
import subprocess
import tszip, tskit
from datetime import datetime
from operator import le
# --- lib --- #

def tag():
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"

# --- implm --- #
infile = snakemake.input.trees
outfile = snakemake.output[0]
with open(snakemake.log.out, "w") as out, open(snakemake.log.err, "w") as err:
    # First, we have to uncompress the SINGER
    ts = tszip.decompress(infile)
    # For now, this code assumes it's using the `shadow` rule
    base = os.path.basename(os.path.splitext(infile)[0])
    temp_file = f"{base}.trees"
    ts.dump(temp_file)
    # Second, we have to create a mutation rate map
    ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
    with open(f"{base}.ratemap", 'w') as f:
        left, right, rates = ratemap.left, ratemap.right, ratemap.rate
        for x, y, z in zip(left, right, rates):
            f.write(f"{int(x)} {int(y)} {z}\n")

    invocation = [
        "python",
        f"{snakemake.params.polegon_binary}",
        "-mutation_map", f"{base}.ratemap",
        "-burn_in", str(snakemake.params.burnin),
        "-num_samples", str(snakemake.params.samples),
        "-scaling_rep", str(snakemake.params.scaling_rep),
        "-thin", str(snakemake.params.thin),
        "-max_step", str(snakemake.params.max_step),
        "-input", f"{base}",
        "-output", f"{base}.polegon"
    ]
    print(f"{tag()}", " ".join(invocation), file=out, flush=True)
    process = subprocess.run(invocation, check=False, stdout=out, stderr=err)
    if process.returncode != 0:
        print(f"{tag()} SINGER run failed ({process.returncode})", file=out, flush=True)
    print(f"{tag()} SINGER run ended ({process.returncode})", file=out, flush=True)
    # Third, we have to compress the SINGER
    out_ts = tskit.load(f"{base}.polegon.trees")
    tszip.compress(out_ts, outfile)
