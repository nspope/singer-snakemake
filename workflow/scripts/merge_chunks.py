"""
Merge chunks into a tree sequence per chromosome and MCMC sample.

Part of https://github.com/nspope/singer-snakemake.
"""

import msprime
import os
import subprocess
import tskit
import pickle
import numpy as np
import yaml
import json
import tszip
from datetime import datetime


# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


def pipeline_provenance(version_string, parameters):
    git_dir = os.path.join(snakemake.scriptdir, os.path.pardir, os.path.pardir, ".git")
    git_commit = subprocess.run(
        ["git", f"--git-dir={git_dir}", "describe", "--always"],
        capture_output=True,
    )
    if git_commit.returncode == 0:
        git_commit = git_commit.stdout.strip().decode('utf-8')
        version_string = f"{version_string}.{git_commit}"
    return {
        "software": {"name": "singer-snakemake", "version": version_string},
        "parameters": snakemake.config,
        "environment": tskit.provenance.get_environment(),
    }


def singer_provenance(version_string, parameters):
    return {
        "software": {"name": "singer", "version": version_string},
        "parameters": parameters,
        "environment": tskit.provenance.get_environment(),
    }


# --- implm --- #

min_branch_length = 1e-7  # TODO: make settable?
stratify = snakemake.params.stratify

ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
metadata = pickle.load(open(snakemake.input.metadata, "rb"))

tables = tskit.TableCollection(sequence_length=ratemap.sequence_length)
nodes, edges, individuals, populations = \
    tables.nodes, tables.edges, tables.individuals, tables.populations

parameters = []
population_map = {}
num_nodes, num_samples = 0, 0
files = zip(snakemake.input.params, snakemake.input.recombs)
for i, (params_file, recomb_file) in enumerate(files):

    node_file = recomb_file.replace("_recombs_", "_nodes_")
    mutation_file = recomb_file.replace("_recombs_", "_muts_")
    branch_file = recomb_file.replace("_recombs_", "_branches_")
    params = yaml.safe_load(open(params_file))
    block_start = params['start']
    parameters.append(params)

    # nodes
    node_time = np.loadtxt(node_file)
    num_nodes = nodes.num_rows - num_samples
    if individuals.num_rows == 0:
        population = []
        num_samples = np.sum(node_time == 0.0)
        individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
        populations.metadata_schema = tskit.MetadataSchema.permissive_json()
        for meta in metadata: 
            individuals.add_row(metadata=meta)
            if stratify in meta:  # recode as integer
                population_name = meta[stratify] 
                if not population_name in population_map:
                    population_map[population_name] = len(population_map)
                    populations.add_row(metadata={"name": population})
                population.append(population_map[population_name])
            else:
                population.append(-1)
        ploidy = num_samples / individuals.num_rows
        assert ploidy == 1.0 or ploidy == 2.0
        for i in range(num_samples):
            individual = i // int(ploidy)
            nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE, 
                population=population[individual],
                individual=individual,
            )
    min_time = 0
    for t in node_time:  # NB: nodes are sorted, ascending in time
        if t > 0.0:
            #TODO: assertion triggers rarely (FP error?)
            #assert t >= min_time 
            t = max(min_time + min_branch_length, t)
            nodes.add_row(time=t)
            min_time = t

    # edges
    edge_span = np.loadtxt(branch_file)
    edge_span = edge_span[edge_span[:, 2] >= 0, :]
    length = max(edge_span[:, 1])
    parent_indices = np.array(edge_span[:, 2], dtype=np.int32)
    child_indices = np.array(edge_span[:, 3], dtype=np.int32)
    parent_indices[parent_indices >= num_samples] += num_nodes
    child_indices[child_indices >= num_samples] += num_nodes
    edges.append_columns(
        left=edge_span[:, 0] + block_start,
        right=edge_span[:, 1] + block_start,
        parent=parent_indices,
        child=child_indices
    )

    # mutations
    mutations = np.loadtxt(mutation_file)
    num_mutations = mutations.shape[0]
    mut_pos = 0
    for i in range(num_mutations):
        if mutations[i, 0] != mut_pos and mutations[i, 0] < length:
            tables.sites.add_row(
                position=mutations[i, 0] + block_start,
                ancestral_state='0',
            )
            mut_pos = mutations[i, 0]
        site_id = tables.sites.num_rows - 1
        mut_node = int(mutations[i, 1])
        if (mut_node < num_samples):
            tables.mutations.add_row(
                site=site_id, 
                node=int(mutations[i, 1]), 
                derived_state=str(int(mutations[i, 3]))
            ) 
        else:
            tables.mutations.add_row(
                site=site_id, 
                node=int(mutations[i, 1]) + num_nodes, 
                derived_state=str(int(mutations[i, 3]))
            )    

# rebuild mutations table in time order at each position
mut_time = tables.nodes.time[tables.mutations.node]
mut_coord = tables.sites.position[tables.mutations.site]
mut_order = np.lexsort((-mut_time, mut_coord))
mut_state = tskit.unpack_strings(
    tables.mutations.derived_state, 
    tables.mutations.derived_state_offset,
)
mut_state, mut_state_offset = tskit.pack_strings(np.array(mut_state)[mut_order])
tables.mutations.set_columns(
    site=tables.mutations.site[mut_order],
    node=tables.mutations.node[mut_order],
    time=np.repeat(tskit.UNKNOWN_TIME, tables.mutations.num_rows),
    derived_state=mut_state,
    derived_state_offset=mut_state_offset,
)

tables.provenances.add_row(
    json.dumps(
        pipeline_provenance(snakemake.params.version["pipeline"], snakemake.config)
    )
)
tables.provenances.add_row(
    json.dumps(
        singer_provenance(snakemake.params.version["singer"], parameters)
    )
)

tables.sort()
tables.build_index()
tables.compute_mutation_parents()
ts = tables.tree_sequence()
tszip.compress(ts, snakemake.output.trees)
