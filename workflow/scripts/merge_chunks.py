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

from utils import absorb_mutations_above_root
from utils import find_genealogical_gaps


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


def tool_provenance(name, version_string, parameters):
    return {
        "software": {"name": name, "version": version_string},
        "parameters": parameters,
        "environment": tskit.provenance.get_environment(),
    }


def force_positive_branch_lengths(nodes_time, edges_parent, edges_child, min_length=1e-7):
    adj_nodes_time = nodes_time.copy()
    #edge_traversal_order = np.argsort(adj_nodes_time[edges_child], kind="stable")
    # assume that parents come after children in node ordering in SINGER output
    edge_traversal_order = np.argsort(edges_child)
    for e in edge_traversal_order:
        p, c = edges_parent[e], edges_child[e]
        if adj_nodes_time[p] - adj_nodes_time[c] < min_length:
            adj_nodes_time[p] = adj_nodes_time[c] + min_length
    assert np.all(adj_nodes_time[edges_parent] - adj_nodes_time[edges_child] > 0)
    return adj_nodes_time



# --- implm --- #

min_branch_length = snakemake.params.min_branch_length
stratify = snakemake.params.stratify

logfile = open(snakemake.log.log, "w")
chunks = pickle.load(open(snakemake.input.chunks, "rb"))
metadata = pickle.load(open(snakemake.input.metadata, "rb"))
alleles = pickle.load(open(snakemake.input.alleles, "rb"))
inaccessible = pickle.load(open(snakemake.input.inaccessible, "rb"))

tables = tskit.TableCollection(sequence_length=chunks.sequence_length)
tables.time_units = "generations"
nodes, edges, individuals, populations = \
    tables.nodes, tables.edges, tables.individuals, tables.populations

tables.metadata_schema = tskit.MetadataSchema.permissive_json()
nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
edges.metadata_schema = tskit.MetadataSchema.permissive_json()
individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
populations.metadata_schema = tskit.MetadataSchema.permissive_json()

singer_parameters = []
polegon_parameters = []
population_map = {}
num_nodes, num_samples = 0, 0
files = zip(
    snakemake.input.params, 
    snakemake.input.recombs, 
    snakemake.input.nodes, 
    snakemake.input.muts, 
    snakemake.input.branches,
)
for i, (params_file, recomb_file, node_file, mutation_file, branch_file) in enumerate(files):

    params = yaml.safe_load(open(params_file))
    block_start = params['singer']['start']
    singer_parameters.append(params['singer'])
    polegon_parameters.append(params['polegon'])

    # nodes
    node_time = np.loadtxt(node_file)
    num_nodes = nodes.num_rows - num_samples
    if individuals.num_rows == 0:
        population = []
        num_samples = np.sum(node_time == 0.0)
        for meta in metadata: 
            individuals.add_row(metadata=meta)
            if stratify in meta:  # recode as integer
                population_name = meta[stratify] 
                if not population_name in population_map:
                    population_map[population_name] = len(population_map)
                    populations.add_row(metadata={"name": population_name})
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
    for t in node_time:
        if t > 0.0:
            nodes.add_row(time=t)

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
            site_pos = block_start + mutations[i, 0]
            site_alleles = alleles[site_pos]
            tables.sites.add_row(
                position=site_pos,
                ancestral_state=site_alleles[0],
            )
            mut_pos = mutations[i, 0]
        site_id = tables.sites.num_rows - 1
        mut_node = int(mutations[i, 1])
        mut_state = int(mutations[i, 3])
        tables.mutations.add_row(
            site=site_id, 
            node=mut_node if mut_node < num_samples else mut_node + num_nodes,
            derived_state=site_alleles[mut_state],
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

# add provenance recording how tree sequence was generated
tables.provenances.add_row(
    json.dumps(
        pipeline_provenance(snakemake.params.version["pipeline"], snakemake.config)
    )
)
tables.provenances.add_row(
    json.dumps(
        tool_provenance(
            "singer", 
            snakemake.params.version["singer"], 
            singer_parameters,
        )
    )
)
tables.provenances.add_row(
    json.dumps(
        tool_provenance(
            "polegon", 
            snakemake.params.version["polegon"], 
            polegon_parameters,
        )
    )
)

# ensure positive branch lengths
branch_lengths = tables.nodes.time[tables.edges.parent] - tables.nodes.time[tables.edges.child]
min_branch = branch_lengths.argmin()
min_obs_branch_length = branch_lengths[min_branch]
if min_obs_branch_length < min_branch_length:
    logfile.write(
        f"{tag()} Minimum branch length is {min_obs_branch_length}, "
        f"forcing minimum branch lengths of {min_branch_length}\n"
    )
    constrained_times = force_positive_branch_lengths(
        tables.nodes.time, 
        tables.edges.parent, 
        tables.edges.child,
    )
    altered_nodes = constrained_times != tables.nodes.time
    assert np.all(constrained_times[altered_nodes] - tables.nodes.time[altered_nodes] > 0)
    max_alteration = (constrained_times[altered_nodes] - tables.nodes.time[altered_nodes]).max()
    logfile.write(
        f"{tag()} Increased ages of {altered_nodes.sum()} nodes, "
        f"with max increase of {max_alteration} generations\n"
    )
    tables.nodes.time = constrained_times

# convert to tree sequence
logfile.write(f"{tag()} Setting mutation ages to branch midpoints\n")
tables.sort()
tables.build_index()
tables.compute_mutation_parents()
tables.compute_mutation_times()
ts = tables.tree_sequence()

# change ancestral state at sites that have had polarisation flipped
if snakemake.params.repolarise:
    prev_num_mutations = ts.num_mutations
    ts = absorb_mutations_above_root(ts)
    logfile.write(
        f"{tag()} Absorbed {prev_num_mutations - ts.num_mutations} mutations "
        f"above the root, switching the ancestral state at these sites\n"
    )

# delete masked intervals that are not spanned by any edge
if snakemake.params.delete_genealogical_gaps:
    genealogical_gaps = find_genealogical_gaps(
        ts,
        inaccessible.position,
        inaccessible.rate == 1.0,
    )
    if genealogical_gaps.size:
        masked_bases = int(np.diff(genealogical_gaps, axis=-1).sum())
        logfile.write(
            f"{tag()} Removing {genealogical_gaps.shape[0]} masked intervals that "
            f"are not spanned by an edge, totalling {masked_bases} bases\n"
        )
        prev_num_edges, prev_num_trees = ts.num_edges, ts.num_trees
        ts = ts.delete_intervals(genealogical_gaps)
        logfile.write(
            f"{tag()} Deleting these genealogical gaps changed the number of edges from "
            f"{prev_num_edges} to {ts.num_edges} and the number of trees from "
            f"{prev_num_trees} to {ts.num_trees}\n"
        )

# delete all masked intervals (for debugging purposes)
if snakemake.params.delete_masked_intervals:
    masked_gaps = np.stack([
        inaccessible.left[inaccessible.rate == 1.0],
        inaccessible.right[inaccessible.rate == 1.0],
    ], axis=-1)
    if masked_gaps.size:
        masked_bases = int(np.diff(masked_gaps, axis=-1).sum())
        logfile.write(
            f"{tag()} Removing {masked_gaps.shape[0]} masked intervals "
            f"totalling {masked_bases} bases\n"
        )
        prev_num_edges, prev_num_trees = ts.num_edges, ts.num_trees
        ts = ts.delete_intervals(masked_gaps)
        logfile.write(
            f"{tag()} Removing all masked intervals changed the number of edges from "
            f"{prev_num_edges} to {ts.num_edges} and the number of trees from "
            f"{prev_num_trees} to {ts.num_trees}\n"
        )

logfile.write(f"{tag()} Merged tree sequence:\n{ts}\n")
tszip.compress(ts, snakemake.output.trees)
