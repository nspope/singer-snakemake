"""
Run POLEGON on a chromosome. Not currently used.

Part of https://github.com/nspope/singer-snakemake.
"""

import os
import yaml
import subprocess
import numpy as np
import tszip
import tskit
from datetime import datetime

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


def root_table(ts):
    left, root = 0.0, tskit.NULL
    stems = []
    for tree in ts.trees():
        right = tree.interval.left
        next_root = tree.root if tree.num_edges else tskit.NULL
        if next_root != root:
            stems.append([left, right, tskit.NULL, root])
            left, root = right, next_root
    stems.append([left, ts.sequence_length, tskit.NULL, root])
    stems = np.array(stems).reshape(-1, 4)
    nonmissing = stems[..., -1] != tskit.NULL
    return stems[nonmissing]


def mutation_table(ts):
    edges = np.array([m.edge for m in ts.mutations()])
    positions = ts.sites_position[ts.mutations_site]
    children = ts.edges_child[edges]
    parents = ts.edges_parent[edges]
    derived_states = np.ones(edges.size)
    mut_table = np.stack([positions, children, parents, derived_states], axis=-1)
    nonmissing = edges != tskit.NULL
    return mut_table[nonmissing]


def write_polegon_inputs(ts, prefix):
    edges = ts.tables.edges
    edge_table = np.stack([edges.left, edges.right, edges.parent, edges.child], axis=-1)
    edge_table = np.concatenate([edge_table, root_table(ts)], axis=0)
    np.savetxt(f"{prefix}_branches.txt", edge_table)
    np.savetxt(f"{prefix}_muts.txt", mutation_table(ts))
    np.savetxt(f"{prefix}_nodes.txt", ts.tables.nodes.time)


# --- implm --- #

use_polegon = True
params = yaml.safe_load(open(snakemake.input.params))
params["seed"] += snakemake.params.seed_offset

params["Ne"] *= 2  
# TODO this will match what singer-master does, but should cancel anyways?

ts = tszip.load(snakemake.input.trees)

if use_polegon:
    write_polegon_inputs(ts, snakemake.input.trees)
    invocation = [
        f"{snakemake.params.polegon_binary}",
        "-input", f"{snakemake.input.trees}",
    ]
    for arg, val in params.items():
        invocation += f"-{arg} {val}".split()
    
    with open(snakemake.log.log, "w") as logfile:
        print(f"{tag()}", " ".join(invocation), file=logfile, flush=True)
        process = subprocess.run(invocation, check=False, stdout=logfile, stderr=logfile)
        print(f"{tag()} POLEGON run ended ({process.returncode})", file=logfile, flush=True)
    assert process.returncode == 0, f"POLEGON terminated with error ({process.returncode})"
    
    tables = ts.dump_tables()
    tables.nodes.time = np.loadtxt(f"{snakemake.input.trees}_new_nodes.txt").flatten()
    tables.sort()
    #tables.provenance.add_row(...) #TODO
    ts = tables.tree_sequence()

tszip.compress(ts, snakemake.output.trees)

#for suffix in ["muts.txt", "branches.txt", "nodes.txt", "new_nodes.txt"]:
#    os.remove(f"{snakemake.input.trees}_{suffix}")
