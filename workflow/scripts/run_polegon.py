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
import pickle
from datetime import datetime

from utils import clip_and_shift_intervals
from utils import remove_partial_ancestry

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
    nonmissing = stems[..., -1] != tskit.NULL
    return stems[nonmissing]


def mutation_table(ts: tskit.TreeSequence) -> tskit.MutationTable:
    edges = np.array([m.edge for m in ts.mutations()])
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
    nonmissing = edges != tskit.NULL
    return mut_table[nonmissing]


def write_polegon_inputs(
    ts: tskit.TreeSequence, 
    nodes_file: str, 
    branches_file: str, 
    mutations_file: str,
) -> None:
    edges = ts.tables.edges
    edge_table = np.stack([edges.left, edges.right, edges.parent, edges.child], axis=-1)
    edge_table = np.concatenate([edge_table, root_table(ts)], axis=0)
    np.savetxt(nodes_file, ts.tables.nodes.time)
    np.savetxt(branches_file, edge_table)
    np.savetxt(mutations_file, mutation_table(ts))


# --- implm --- #

logfile = open(snakemake.log.log, "w")
use_polegon = snakemake.params.use_polegon
use_mutational_span = snakemake.params.use_mutational_span
drop_omitted = snakemake.params.drop_omitted
use_node_masks = snakemake.params.use_node_masks
use_deprecated = snakemake.params.use_deprecated  # TODO: remove when ready
node_masks = pickle.load(open(snakemake.input.node_masks, "rb"))
failed_chunk = os.path.getsize(snakemake.input.nodes) == 0

if use_deprecated and use_polegon and not failed_chunk:
    # NOTE: this is the old code path, retained for reference and testing.
    # FIXME: Remove once convinced this new code path is equivalent sans sample masks.

    singer_params = yaml.safe_load(open(snakemake.input.params)).pop("singer")
    polegon_params = yaml.safe_load(open(snakemake.input.params)).pop("polegon")
    seed = polegon_params.pop("seed") + int(snakemake.wildcards.rep)

    # FIXME: Ne is lower by factor of two relative to `polegon_master`.
    # This shouldn't matter, it cancels during rescaling, but look into it.

    # POLEGON expects inputs named slightly differently than SINGER output
    prefix = snakemake.input.muts.replace("_muts_", "_").removesuffix(".txt")
    shutil.copy(snakemake.input.nodes, f"{prefix}_nodes.txt")

    # Adjust mutation list to omit specified sites. These will not be used for
    # dating, and are effectively inaccessible sequence. However, they were used to
    # build the topologies and will be retained in the trees.
    if drop_omitted:
        # FIXME: for a fully correct clock, the bases occupied by omitted sites
        # should be subtracted from edge spans-- but if we assume omitted sites
        # are a very small proportion of edge spans, then the bias should be
        # minimal.
        left, right = singer_params["start"], singer_params["end"]
        omitted_positions = pickle.load(open(snakemake.input.omitted, "rb"))
        in_bounds = np.logical_and(omitted_positions >= left, omitted_positions < right)
        omitted_positions = omitted_positions[in_bounds] - left
        logfile.write(f"{tag()} Read list of {omitted_positions.size} omitted positions\n")
        mutations = np.loadtxt(snakemake.input.muts)
        omitted_mutations = np.isin(mutations[:, 0].astype(np.int64), omitted_positions)
        logfile.write(f"{tag()} Omitting {omitted_mutations.sum()} mutations from dating\n")
        np.savetxt(f"{prefix}_muts.txt", mutations[~omitted_mutations])
    else:
        shutil.copy(snakemake.input.muts, f"{prefix}_muts.txt")

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
        polegon_params["m"] = 1.0
        mutation_map = polegon_params.pop("mutation_map")
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
    for arg, val in polegon_params.items():
        invocation += f"-{arg} {val}".split()
    
    logfile.write(f"{tag()} " + " ".join(invocation) + "\n")
    process = subprocess.run(invocation, check=False, stdout=logfile, stderr=logfile)
    logfile.write(f"{tag()} POLEGON run ended ({process.returncode})\n")

    for suffix in ["muts", "branches", "nodes"]:
        os.remove(f"{prefix}_{suffix}.txt")

    assert process.returncode == 0, f"POLEGON terminated with error ({process.returncode})"
    os.rename(f"{prefix}_new_nodes.txt", snakemake.output.nodes)
elif not use_deprecated and use_polegon and not failed_chunk:
    # This is the new code path: a major rewrite. It should give the same output as the
    # original code path provided no sample masks are provided. The crucial assumption is
    # that the node table is kept invariant throughout.
    singer_params = yaml.safe_load(open(snakemake.input.params)).pop("singer")
    polegon_params = yaml.safe_load(open(snakemake.input.params)).pop("polegon")
    seed = polegon_params.pop("seed") + int(snakemake.wildcards.rep)
    left, right = singer_params["start"], singer_params["end"]
    prefix = snakemake.input.muts.replace("_muts_", "_").removesuffix(".txt")

    treeseq = singer_to_tree_sequence(
        snakemake.input.nodes, 
        snakemake.input.branches, 
        snakemake.input.muts,
    )
    logfile.write(f"{tag()} Read tree sequence from SINGER input:\n{treeseq}\n")

    if node_masks and use_node_masks:
        node_masks = {
            sample: clip_and_shift_intervals(intervals, left, right)
            for sample, intervals in node_masks.items()
        }
        treeseq = remove_partial_ancestry(treeseq, node_masks, filter_nodes=False)
        logfile.write(f"{tag()} Trimmed tree sequence given per-sample masks:\n{treeseq}\n")

    # Write out POLEGON inputs
    polegon_nodes = f"{prefix}_nodes.txt"
    polegon_branches = f"{prefix}_branches.txt"
    polegon_muts = f"{prefix}_muts.txt"
    write_polegon_inputs(treeseq, polegon_nodes, polegon_branches, polegon_muts)

    # Adjust mutation list to omit specified sites. These will not be used for
    # dating, and are effectively inaccessible sequence. However, they were used to
    # build the topologies and will be retained in the trees.
    if drop_omitted:
        # FIXME: for a fully correct clock, the bases occupied by omitted sites
        # should be subtracted from edge spans-- but if we assume omitted sites
        # are a very small proportion of edge spans, then the bias should be
        # minimal.
        omitted_positions = pickle.load(open(snakemake.input.omitted, "rb"))
        in_bounds = np.logical_and(omitted_positions >= left, omitted_positions < right)
        omitted_positions = omitted_positions[in_bounds] - left
        logfile.write(f"{tag()} Read list of {omitted_positions.size} omitted positions\n")
        mutations = np.loadtxt(polegon_muts)
        omitted_mutations = np.isin(mutations[:, 0].astype(np.int64), omitted_positions)
        logfile.write(f"{tag()} Omitting {omitted_mutations.sum()} mutations from dating\n")
        np.savetxt(polegon_muts, mutations[~omitted_mutations])

    # Adjust branch spans to reflect masked sequence, and absorb mutation rate
    # into span.  This is necessary because POLEGON doesn't use the "average
    # branch mutation rate" from the mutation map during rescaling, and instead
    # just adjusts by the mean rate. Manually converting the branch spans to
    # mutational units fixes this.
    if use_mutational_span:
        logfile.write(
            f"{tag()} Adjusting branch spans input ({polegon_branches}) "
            f"to reflect mutation rate map and setting mutation rate to unity\n"
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
        branches = np.loadtxt(polegon_branches)
        for i in range(2):
           branches[:, i] = adjusted_mu.get_cumulative_mass(branches[:, i])
        np.savetxt(polegon_branches, branches)

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

    for file in [polegon_nodes, polegon_branches, polegon_muts]: os.remove(file)

    assert process.returncode == 0, f"POLEGON terminated with error ({process.returncode})"
    os.rename(f"{prefix}_new_nodes.txt", snakemake.output.nodes)
else:
    logfile.write(f"{tag()} Skipping dating with POLEGON\n")
    shutil.copy(snakemake.input.nodes, snakemake.output.nodes)
