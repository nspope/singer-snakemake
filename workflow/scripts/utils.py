"""
Utility functions for converting between formats and manipulating tree sequences.

Part of https://github.com/nspope/singer-snakemake.
"""

import numpy as np
import msprime
import tskit


def bitmask_to_arrays(bitmask: np.ndarray, *, insert_breakpoints: np.ndarray = None) -> msprime.RateMap:
    """
    Convert a bitmask to a binary interval mask, with additional
    `insert_breakpoints` if not `None`. 
    """
    assert np.issubdtype(bitmask.dtype, bool)
    changepoints = np.flatnonzero(bitmask[1:] != bitmask[:-1]) + 1
    changepoints = np.append(np.append(0, changepoints), bitmask.size)
    if insert_breakpoints is not None:
        assert np.issubdtype(insert_breakpoints.dtype, int)
        assert insert_breakpoints.min() >= 0 and insert_breakpoints.max() <= bitmask.size
        changepoints = \
            np.unique(np.append(insert_breakpoints, changepoints))
    values = bitmask[changepoints[:-1]].astype(float)
    return values, changepoints


def ratemap_to_text(ratemap: msprime.RateMap, *, replace_nan_with: float = 0.0) -> str:
    """
    Write a ratemap to a headerless, three-column text file that is
    left coordinate, right coordinate, per-base rate
    """
    text = []
    for left, right, rate in zip(
        ratemap.position[:-1],  # FIXME: use ratemap.left/right
        ratemap.position[1:], 
        ratemap.rate,
    ):
        if np.isnan(rate): rate = replace_nan_with
        text.append(f"{int(left)} {int(right)} {rate:.16f}") # FIXME: precision FIXME: no int
    text = "\n".join(text) + "\n"
    return text 


def absorb_mutations_above_root(ts: tskit.TreeSequence) -> tskit.TreeSequence:
    """
    Remove mutations above the root, and change the ancestral state of the
    site to match the state of the root node. Compute times for remaining
    mutations
    """
    tab = ts.dump_tables()
    tab.mutations.clear()
    tab.sites.clear()
    # mutations are sorted with "oldest" mutation parent first, so take new
    # ancestral state from last mutation above root. (further, if trees are not
    # simplified, there should be only one such mutation per site in SINGER
    # output).
    tree = ts.first()
    for s in ts.sites():
        tree.seek(s.position)
        assert len(tree.roots) == 1
        ancestral_state = s.ancestral_state
        for m in s.mutations:
            if m.node == tree.root:
                ancestral_state = m.derived_state
            else:
                tab.mutations.add_row(
                    node=m.node, 
                    time=m.time,
                    site=m.site,
                    metadata=m.metadata,
                    derived_state=m.derived_state,
                )
        tab.sites.add_row(
            position=s.position,
            metadata=s.metadata,
            ancestral_state=ancestral_state,
        )
    tab.sort()
    tab.build_index()
    tab.compute_mutation_parents()
    out = tab.tree_sequence()
    assert out.num_sites == ts.num_sites
    return out


def mutational_load(ts: tskit.TreeSequence, windows: np.ndarray = None) -> np.ndarray:
    """
    TODO
    """
    genome_windows = np.array([0, ts.sequence_length]) if windows is None else windows
    assert genome_windows[0] == 0 and genome_windows[-1] == ts.sequence_length
    mutations_window = np.digitize(ts.sites_position[ts.mutations_site], genome_windows) - 1
    assert mutations_window.min() >= 0 and mutations_window.max() < genome_windows.size - 1
    load = np.zeros((genome_windows.size - 1, ts.num_samples))
    tree = ts.first(sample_lists=True)
    for s in ts.sites():
        tree.seek(s.position)
        for m in s.mutations:
            if m.edge != tskit.NULL:
                window = mutations_window[m.id]
                samples = list(tree.samples(m.node))
                load[window, samples] += 1.0
    return load.squeeze(0) if windows is None else load


def collapse_masked_intervals(ts: tskit.TreeSequence, accessible: msprime.RateMap) -> tskit.TreeSequence:
    """
    Return a copy of the tree sequence with masked intervals (where `accessible.rate == 0.0`)
    collapsed, so that the coordinate system is in terms of unmasked sequence length.
    Zero length edges are removed, and any nodes that are then disconnected are removed as well.
    All sites and mutations that are within the collapsed intervals are removed.
    """
    assert np.logical_or(accessible.rate == 0.0, accessible.rate == 1.0)
    assert accessible.sequence_length == ts.sequence_length
    tab = ts.dump_tables()
    # map edges to new coordinate system and remove those with zero length
    tab.edges.left = accessible.get_cumulative_mass(tab.edges.left)
    tab.edges.right = accessible.get_cumulative_mass(tab.edges.right)
    tab.edges.keep_rows(tab.edges.right > tab.edges.left)
    # remove disconnected nodes
    is_connected = np.full(tab.nodes.num_rows, False)
    is_connected[tab.edges.parent] = True
    is_connected[tab.edges.child] = True
    node_map = tab.nodes.keep_rows(is_connected)
    # map sites to new coordinate system and remove those in masked intervals
    site_map = tab.sites.keep_rows(accessible.get_rate(tab.sites.position).astype(bool))
    tab.sites.position = accessible.get_cumulative_mass(tab.sites.position)
    # update mutation pointers and remove those without a node or site
    tab.mutations.node = node_map[tab.mutations.node]
    tab.mutations.site = site_map[tab.mutations.site]
    tab.mutations.keep_rows(np.logical_and(tab.mutations.site != tskit.NULL, tab.mutations.node != tskit.NULL))
    tab.sort()
    tab.build_index()
    tab.compute_mutation_parents()
    return tab.tree_sequence()


def find_genealogical_gaps(ts: tskit.TreeSequence, accessible: msprime.RateMap) -> np.ndarray:
    """
    Find masked intervals (where `accessible.rate == 0.0`) that do not have
    one or edges extending over. In other words, all edges entering the masked interval
    must end inside the masked interval.
    """
    pass
