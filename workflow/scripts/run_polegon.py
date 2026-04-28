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
import tskit
import pickle
from datetime import datetime

from utils import clip_and_shift_intervals
from utils import effective_edge_spans

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# TODO: clean this up. In particular compute_mutation_parents should take care
# of the sort with latest tskit.
def singer_to_tree_sequence(
    node_file: str, 
    edge_file: str, 
    mutation_file: str,
) -> tskit.TreeSequence:
    node_time = np.loadtxt(node_file)
    edge_span = np.loadtxt(edge_file)
    mutations = np.loadtxt(mutation_file)

    # edges and nodes
    edge_span = edge_span[edge_span[:, 2] >= 0, :]
    length = max(edge_span[:, 1])

    tables = tskit.TableCollection(sequence_length=length)
    node_table = tables.nodes
    edge_table = tables.edges
    for t in node_time:
        if (t == 0):
            node_table.add_row(flags=tskit.NODE_IS_SAMPLE)
        else:
            node_table.add_row(time=t)
    parent_indices = np.array(edge_span[:, 2], dtype=np.int32)
    child_indices = np.array(edge_span[:, 3], dtype=np.int32)
    edge_table.set_columns(
        left=edge_span[:, 0], 
        right=edge_span[:, 1], 
        parent=parent_indices, 
        child=child_indices,
    )

    # mutations
    n = mutations.shape[0]
    mut_pos = -1
    site_id = 0
    for i in range(n):
        if mutations[i, 0] != mut_pos and mutations[i, 0] < tables.sequence_length:
            tables.sites.add_row(position=mutations[i, 0], ancestral_state='0')
            mut_pos = mutations[i, 0]
        site_id = tables.sites.num_rows - 1
        tables.mutations.add_row(site=site_id, node=int(mutations[i, 1]), derived_state=str(int(mutations[i, 3])))
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

    # finalize
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    tables.compute_mutation_times()
    return tables.tree_sequence()


def root_table(ts: tskit.TreeSequence) -> np.ndarray:
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
    return stems


def mutation_table(ts: tskit.TreeSequence) -> np.ndarray:
    edges = ts.mutations_edge
    positions = ts.sites_position[ts.mutations_site]
    children = ts.edges_child[edges]
    parents = ts.edges_parent[edges]
    derived_states = np.array([
        int(s) for s in tskit.unpack_strings(
            ts.tables.mutations.derived_state,
            ts.tables.mutations.derived_state_offset,
        )
    ])
    mut_table = np.stack([positions, children, parents, derived_states], axis=-1)
    return mut_table



# --- implm --- #

logfile = open(snakemake.log.log, "w")
use_polegon = snakemake.params.use_polegon
use_mutational_span = snakemake.params.use_mutational_span
remove_polegon_inputs = snakemake.params.remove_polegon_inputs
drop_omitted = snakemake.params.drop_omitted
skip_node_masks = snakemake.params.skip_node_masks
node_masks = pickle.load(open(snakemake.input.node_masks, "rb"))
failed_chunk = os.path.getsize(snakemake.input.nodes) == 0

if use_polegon and not failed_chunk:
    params_yaml = yaml.safe_load(open(snakemake.input.params))
    singer_params = params_yaml.pop("singer")
    polegon_params = params_yaml.pop("polegon")
    seed = polegon_params.pop("seed") + int(snakemake.wildcards.rep)
    start, end = singer_params["start"], singer_params["end"]
    prefix = snakemake.input.muts.replace("_muts_", "_").removesuffix(".txt")

    treeseq = singer_to_tree_sequence(
        snakemake.input.nodes, 
        snakemake.input.branches, 
        snakemake.input.muts,
    )
    logfile.write(f"{tag()} Read tree sequence from SINGER input:\n{treeseq}\n")

    # Absorb mutation rate into span.  This is necessary because POLEGON
    # doesn't use the "average branch mutation rate" from the mutation map
    # during rescaling, and instead just adjusts by the mean rate. Manually
    # converting the branch spans to mutational units fixes this.
    if use_mutational_span:
        logfile.write(
            f"{tag()} Adjusting coordinate system to reflect mutation rate map "
            f"and setting mutation rate to unity\n"
        )
        polegon_params["m"] = 1.0
        mutation_map = polegon_params.pop("mutation_map")
        # TODO: it would be easier to load in the mut_rate pickle and clip it
        adjusted_mu = np.loadtxt(mutation_map, ndmin=2)
        assert np.all(adjusted_mu[:-1, 1] == adjusted_mu[1:, 0])
        adjusted_mu = msprime.RateMap(
           position=np.append(adjusted_mu[0, 0], adjusted_mu[:, 1]),
           rate=adjusted_mu[:, 2],
        )
    else:
        adjusted_mu = msprime.RateMap(position=[0, treeseq.sequence_length], rate=[1.0])

    node_masks = None if skip_node_masks else {
        sample: clip_and_shift_intervals(intervals, start, end)
        for sample, intervals in node_masks.items()
    }
    retained_edge_span, retained_mutations = effective_edge_spans(treeseq, adjusted_mu, node_masks)
    logfile.write(
        f"{tag()} Retained {retained_mutations.sum()} of {retained_mutations.size} "
        f"mutations for after masking\n"
    )

    # Adjust mutation list to omit specified sites. These will not be used for
    # dating, and are effectively inaccessible sequence. However, they were used to
    # build the topologies and will be retained in the trees.
    if drop_omitted:
        # FIXME: for a fully correct clock, the bases occupied by omitted sites
        # should be subtracted from edge spans-- but if we assume omitted sites
        # are a very small proportion of edge spans, then the bias should be
        # minimal.
        omitted_positions = pickle.load(open(snakemake.input.omitted, "rb"))
        in_bounds = np.logical_and(omitted_positions >= start, omitted_positions < end)
        omitted_positions = omitted_positions[in_bounds] - start
        omitted_mutations = np.isin(treeseq.sites_position[treeseq.mutations_site], omitted_positions)
        logfile.write(f"{tag()} Omitting {omitted_mutations.sum()} mutations from dating\n")
        retained_mutations = np.logical_and(retained_mutations, ~omitted_mutations)

    # Assemble POLEGON inputs. POLEGON only cares about edge boundaries insofar
    # as these impact edge span.
    retained_mutations = np.logical_and(retained_mutations, treeseq.mutations_edge != tskit.NULL)
    edge_left = adjusted_mu.get_cumulative_mass(treeseq.edges_left)
    edge_right = edge_left + retained_edge_span
    edge_table = np.column_stack([edge_left, edge_right, treeseq.edges_parent, treeseq.edges_child])
    root_stems = root_table(treeseq)
    root_stems[:, :2] = adjusted_mu.get_cumulative_mass(root_stems[:, :2])
    root_stems = root_stems[root_stems[..., -1] != tskit.NULL]
    branches = np.concatenate([edge_table, root_stems], axis=0)
    mutations = mutation_table(treeseq)[retained_mutations]
    nodes = treeseq.nodes_time

    polegon_nodes = f"{prefix}_nodes.txt"
    polegon_branches = f"{prefix}_branches.txt"
    polegon_muts = f"{prefix}_muts.txt"
    np.savetxt(polegon_nodes, nodes)
    np.savetxt(polegon_branches, branches)
    np.savetxt(polegon_muts, mutations)

    invocation = [
        f"{snakemake.params.polegon_binary}",
        "-input", f"{prefix}",
        "-seed", f"{seed}",
    ]
    for arg, val in polegon_params.items():
        invocation += f"-{arg} {val}".split()
    
    logfile.write(f"{tag()} " + " ".join(invocation) + "\n")
    process = subprocess.run(invocation, check=False, stdout=logfile, stderr=logfile)
    logfile.write(f"{tag()} POLEGON run ended ({process.returncode})\n")

    for file in [polegon_nodes, polegon_branches, polegon_muts]: 
        if remove_polegon_inputs: os.remove(file)

    assert process.returncode == 0, f"POLEGON terminated with error ({process.returncode})"
    os.rename(f"{prefix}_new_nodes.txt", snakemake.output.nodes)
else:
    logfile.write(f"{tag()} Skipping dating with POLEGON\n")
    shutil.copy(snakemake.input.nodes, snakemake.output.nodes)
