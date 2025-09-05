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
params = yaml.safe_load(open(snakemake.input.params)).pop("singer")
seed = params.pop("seed")

invocation = [f"{snakemake.params.singer_binary}"]
for arg, val in params.items():
    invocation += f"-{arg} {val}".split()

with open(snakemake.log.log, "w") as log:
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
        log.write(f"{tag()} " + " ".join(invocation + args) + "\n")
        process = subprocess.run(invocation + args, check=False, stdout=log, stderr=log)
        if process.returncode == 0: 
            break
    log.write(f"{tag()} SINGER run ended ({process.returncode})\n")
assert process.returncode == 0, f"SINGER terminated with error ({process.returncode})"
