"""
Run SINGER on a chunk.

Part of https://github.com/nspope/singer-snakemake.
"""

import os
import yaml
import subprocess
from datetime import datetime

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"

# --- implm --- #

logfile = snakemake.input.params.replace(".yaml", ".log")
params = yaml.safe_load(open(snakemake.input.params))
seed = params.pop("seed")

invocation = [f"{snakemake.params.singer_binary}"]
for arg, val in params.items():
    invocation += f"-{arg} {val}".split()

with open(snakemake.log.out, "w") as out, open(snakemake.log.err, "w") as err:
    for attempt in range(snakemake.params.mcmc_resumes):
        args = ["-seed", str(seed + attempt)]
        if os.path.exists(logfile):
            contents = open(logfile, "r").readlines()
            if len(contents) > 1 and "rethread" in contents[-2]:
                # restarting from the current iteration sometimes fails,
                # so restart from the previous iteration instead
                handle = open(logfile, "w")
                handle.write("".join(contents[:-1]))
                handle.close()
                args.append("-resume")
        print(f"{tag()}", " ".join(invocation + args), file=out, flush=True)
        process = subprocess.run(invocation + args, check=False, stdout=out, stderr=err)
        if process.returncode == 0: 
            break
    print(f"{tag()} SINGER run ended ({process.returncode})", file=out, flush=True)

assert process.returncode == 0, f"SINGER terminated with error ({process.returncode})"
