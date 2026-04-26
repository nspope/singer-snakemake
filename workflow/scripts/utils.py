"""
Utility functions for converting between formats and manipulating tree sequences.

Part of https://github.com/nspope/singer-snakemake.
"""

import numpy as np
import msprime
import typing
import tskit
from collections import defaultdict


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
        ratemap.left,
        ratemap.right,
        ratemap.rate,
    ):
        if np.isnan(rate): rate = replace_nan_with
        text.append(f"{int(left)} {int(right)} {rate:.16f}") # FIXME: precision FIXME: no int
    text = "\n".join(text) + "\n"
    return text 


def read_single_fasta(handle: typing.IO) -> None:
    """
    Read a single-contig fasta
    """
    sequence = handle.read().strip().split("\n")
    headers = np.char.startswith(sequence, ">")
    assert len(headers) and headers[0] and np.all(~headers[1:]), "Fasta must contain a single sequence"
    sequence = "".join(sequence[1:])
    return np.fromiter(sequence, dtype="<U1")


def write_minimal_vcf(handle, sample_names, CHROM, POS, ID, REF, ALT, GT): 
    """
    Write a minimal biallelic diploid VCF
    """
    assert CHROM.size == POS.size == ID.size == REF.size
    assert ALT.ndim == 1 and ALT.size == CHROM.size
    assert GT.shape[0] == CHROM.size and GT.shape[1] == sample_names.size and GT.shape[2] == 2
    assert np.all(np.diff(POS) > 0), "Positions non-increasing in VCF"
    handle.write("##fileformat=VCFv4.2\n")
    handle.write("##source=\"singer-snakemake::chunk_chromosomes\"\n")
    handle.write("##FILTER=<ID=PASS,Description=\"All filters passed\">\n")
    handle.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
    handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
    for sample in sample_names: handle.write(f"\t{sample}")
    handle.write("\n")
    for chrom, pos, id, ref, alt, gt in zip(CHROM, POS, ID, REF, ALT, GT):
        handle.write(f"{chrom}\t{pos}\t{id}\t{ref}\t{alt}\t.\tPASS\t.\tGT")
        for (a, b) in gt: handle.write(f"\t{a}|{b}")
        handle.write("\n")


def absorb_mutations_above_root(
    ts: tskit.TreeSequence, 
    record_flipped: bool = True,
) -> tskit.TreeSequence:
    """
    Remove mutations above the root, and change the ancestral state of the
    site to match the state of the root node. 
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
        metadata = s.metadata
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
        if record_flipped:
            metadata["flipped"] = ancestral_state != s.ancestral_state
        tab.sites.add_row(
            position=s.position,
            metadata=metadata,
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
    Calculate the number of derived mutations per sample.
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


def compactify_run_length_encoding(breaks: np.ndarray, values: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Combine adjacent runs in a run length encoding if they have the same value
    """
    assert breaks.size == values.size + 1
    left, right = breaks[:-1], breaks[1:]
    changepoints = np.append(True, values[1:] != values[:-1])
    new_breaks = np.append(left[changepoints], right[-1])
    new_values = values[changepoints]
    return new_breaks, new_values


def find_genealogical_gaps(
    ts: tskit.TreeSequence, 
    interval_breakpoints: np.ndarray, 
    interval_is_gap: np.ndarray,
) -> np.ndarray:
    """
    Find intervals between `interval_breakpoints` where `interval_is_gap`, that
    are not completely contained within any edge.
    """
    assert np.all(np.diff(interval_breakpoints) > 0)
    assert interval_is_gap.size == interval_breakpoints.size - 1
    assert interval_breakpoints[0] == 0.0 and interval_breakpoints[-1] == ts.sequence_length
    # reduce to minimal set of intervals given boolean array
    interval_breakpoints, interval_is_gap = \
        compactify_run_length_encoding(interval_breakpoints, interval_is_gap)
    # TODO: ^^^ move to outside function?
    assert np.all(np.abs(np.diff(interval_is_gap)) == 1.0)  # no adjacent intervals with same value
    interval_left, interval_right = interval_breakpoints[:-1], interval_breakpoints[1:]
    num_intervals = interval_is_gap.size
    # sort edges by left endpoint
    sort_order = np.argsort(ts.edges_left) #ts.edge_insertion_order
    edges_left = ts.edges_left[sort_order]
    edges_right = ts.edges_right[sort_order]
    # find the edges_left immediately before each interval start
    closest_edge = np.digitize(interval_left, edges_left, right=True) - 1
    assert np.all(np.logical_or(closest_edge < 0, edges_left[closest_edge] < interval_left))
    edges_right_max = np.maximum.accumulate(edges_right)
    no_spanning_edge = np.logical_or(
        closest_edge < 0,
        np.logical_and(closest_edge >= 0, edges_right_max[closest_edge] <= interval_right)
    )
    genealogical_gaps = np.logical_and(no_spanning_edge, interval_is_gap)
    intervals = np.stack([interval_left[genealogical_gaps], interval_right[genealogical_gaps]], axis=-1)
    return intervals


def multiply_ratemaps(ratemap: msprime.RateMap, other: msprime.RateMap) -> msprime.RateMap:
    """
    Create a new set of intervals from the intersection of two ratemaps, and then take the 
    product of the rates in each interval.
    """
    assert ratemap.sequence_length == other.sequence_length
    new_position = np.unique(np.append(ratemap.position, other.position))
    assert new_position[0] == 0.0 and new_position[-1] == ratemap.sequence_length
    new_rate = ratemap.get_rate(new_position[:-1]) * other.get_rate(new_position[:-1])
    new_ratemap = msprime.RateMap(position=new_position, rate=new_rate)
    return new_ratemap


def extract_accessible_ratemap(ts: tskit.TreeSequence) -> msprime.RateMap:
    """
    Return a ratemap where the rate is zero over masked segments in the tree
    sequence, and one otherwise.
    """
    breakpoints = ts.breakpoints(as_array=True)
    accessible = np.array([t.num_edges > 0 for t in ts.trees()])
    new_breakpoints, new_accessible = compactify_run_length_encoding(breakpoints, accessible)
    ratemap = msprime.RateMap(position=new_breakpoints, rate=new_accessible)
    return ratemap


def merge_intervals(intervals: np.ndarray) -> np.ndarray:
    """
    Sort and merge overlapping or adjacent intervals.
    """
    if intervals.shape[0] < 2: return intervals
    order = np.argsort(intervals[:, 0], kind="stable")
    left, right = intervals[order].T
    # interval i is a new group if left[i] > max_right[i-1]
    max_right = np.maximum.accumulate(right)
    new_group = np.append(True, left[1:] > max_right[:-1])
    group_id = np.cumsum(new_group) - 1
    num_groups = group_id[-1] + 1
    # merged left is first left, merged right is max right, per group
    merged_left = left[new_group]
    merged_right = np.zeros(num_groups, dtype=right.dtype)
    np.maximum.at(merged_right, group_id, right)
    return np.column_stack([merged_left, merged_right])


def parse_sample_bedmask(
    bed_path: str,
    sample_map: dict[str, int],
) -> dict[int, np.ndarray]:
    """
    Load per-sample mask intervals from a BED file, separate by sample, and sort and merge. 
    The fourth column should be the sample name. `sample_map` maps the sample
    name to the integer ids used to key the output.
    """
    # collect intervals by sample id
    unprocessed_intervals = defaultdict(list)
    with open(bed_path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            assert len(fields) == 4, "Sample-specific bedmask must have four columns"
            _, start, end, sample_name = fields
            if sample_name not in sample_map: continue
            sample_id = sample_map[sample_name]
            unprocessed_intervals[sample_id].append((float(start), float(end)))
    # sort and merge intervals
    intervals_by_sample = {}
    for sample_id, pairs in unprocessed_intervals.items():
        intervals = merge_intervals(np.array(pairs))
        if intervals.shape[0] > 0:
            intervals_by_sample[sample_id] = intervals
    return intervals_by_sample


def clip_and_shift_intervals(
    intervals: np.ndarray,
    start: float = 0.0,
    end: float = np.inf,
    shift: bool = True,
) -> np.ndarray:
    """
    Intersect `intervals` with `[start, end)`. If `shift`, then the coordinate
    system will start at zero.
    """
    left, right = intervals.T
    overlaps = np.minimum(right, end) - np.maximum(left, start) > 0
    intervals = np.clip(intervals[overlaps], start, end)
    if shift: intervals -= start
    return intervals


def interval_coverage(
    intervals_by_sample: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find unique intervals and count coverage. Intervals are left-inclusive.
    """
    assert intervals_by_sample, "Must have at least one set of intervals to intersect"
    starts, ends = np.concatenate(list(intervals_by_sample.values())).T
    positions = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones_like(starts, dtype=int), -np.ones_like(ends, dtype=int)])
    order = np.lexsort((deltas, positions))
    breakpoints = positions[order]
    counts = np.cumsum(deltas[order])
    starts = breakpoints[:-1]
    ends = breakpoints[1:]
    counts = counts[:-1]
    valid = ends > starts
    intervals = np.stack([starts, ends], axis=-1)[valid]
    coverage = counts[valid]
    return intervals, coverage


def remove_partial_ancestry(
    ts: tskit.TreeSequence,
    intervals_by_sample: dict[int, np.ndarray],
    filter_nodes: bool = True,
) -> tskit.TreeSequence:
    """
    Remove ancestry given "masked" `intervals_by_sample`, leaving subtrees.
    These are simplified such that there are no dangling nodes or mutations.
    If `filter_nodes` then the node table is left intact, maintaing IDs.
    """
    edge_order = np.argsort(ts.edges_child, kind="stable")
    node_index = np.searchsorted(ts.edges_child[edge_order], np.arange(ts.num_nodes + 1))
    drop_edge = np.full(ts.num_edges, False)
    drop_muts = np.full(ts.num_mutations, False)

    new_left, new_right, new_parent, new_child = [], [], [], []
    for sample, intervals in intervals_by_sample.items():
        edge_ids = edge_order[node_index[sample]:node_index[sample + 1]]
        if edge_ids.size == 0: continue  # sample is missing
        if intervals.size == 0: continue  # nothing to mask
        assert np.all(np.diff(intervals.flatten()) >= 0)
        interval_left, interval_right = intervals.T
        # filter edges
        for e in edge_ids:
            edge_left, edge_right, edge_parent = \
                ts.edges_left[e], ts.edges_right[e], ts.edges_parent[e]
            # overlaps where interval_left < edge_right and interval_right > edge_left
            first = np.searchsorted(interval_right, edge_left, side="right")
            last = np.searchsorted(interval_left, edge_right, side="left")
            if first < last:
                # traverse overlapping intervals, outputting the gaps between them
                drop_edge[e] = True
                position = edge_left
                for k in range(first, last):
                    clipped_left = max(interval_left[k], edge_left)
                    clipped_right = min(interval_right[k], edge_right)
                    if position < clipped_left:
                        new_left.append(position)
                        new_right.append(clipped_left)
                        new_parent.append(edge_parent)
                        new_child.append(sample)
                    position = clipped_right
                if position < edge_right:
                    new_left.append(position)
                    new_right.append(edge_right)
                    new_parent.append(edge_parent)
                    new_child.append(sample)
        # filter mutations; we only have to worry about mutations above
        # a sample node, as these will be retained by simplify
        sample_muts = np.flatnonzero(ts.mutations_node == sample)
        if sample_muts.size > 0:
            muts_position = ts.sites_position[ts.mutations_site[sample_muts]]
            muts_right = np.searchsorted(interval_right, muts_position, side="right")
            muts_left = np.searchsorted(interval_left, muts_position, side="right") - 1
            muts_inside = muts_left == muts_right
            drop_muts[sample_muts[muts_inside]] = True

    tables = ts.dump_tables()
    tables.mutations.keep_rows(~drop_muts)
    tables.edges.keep_rows(~drop_edge)
    tables.edges.append_columns(
        left=np.asarray(new_left, dtype=tables.edges.left.dtype),
        right=np.asarray(new_right, dtype=tables.edges.right.dtype),
        parent=np.asarray(new_parent, dtype=tables.edges.parent.dtype),
        child=np.asarray(new_child, dtype=tables.edges.child.dtype),
    )
    tables.edges.squash()
    tables.sort()
    tables.simplify(keep_unary=True, filter_nodes=filter_nodes)
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


# TODO: clean up
def adjust_edge_spans_for_partial_ancestry(
    ts: tskit.TreeSequence,
    intervals_by_sample: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply `remove_partial_ancestry` to the tree sequence, then find the
    surviving genomic span for each edge, and the positions and nodes of
    removed mutations.
    """
    masked_ts = remove_partial_ancestry(ts, intervals_by_sample, filter_nodes=False)
    # group edges by parent-child combination
    edge_group = lambda p, c: p.astype(np.int64) * ts.num_nodes + c.astype(np.int64)
    # find group start and end for every masked edge
    masked_group = edge_group(masked_ts.edges_parent, masked_ts.edges_child)
    masked_order = np.lexsort((masked_ts.edges_left, masked_group))
    masked_group = masked_group[masked_order]
    masked_left = masked_ts.edges_left[masked_order]
    masked_right = masked_ts.edges_right[masked_order]
    unique_group, group_start = np.unique(masked_group, return_index=True)
    group_end = np.append(group_start[1:], len(masked_group))
    # compute overlap with masked edges for each original edge
    original_group = edge_group(ts.edges_parent, ts.edges_child)
    pruned_span = np.zeros(ts.num_edges, dtype=np.float64)
    for e in ts.edges():
        key = original_group[e.id]
        idx = np.searchsorted(unique_group, key)
        if idx >= len(unique_group) or unique_group[idx] != key:
            continue  # group no longer exists after pruning
        lo, hi = group_start[idx], group_end[idx]
        left, right = masked_left[lo:hi], masked_right[lo:hi]
        overlap = np.minimum(right, e.right) - np.maximum(left, e.left)
        pruned_span[e.id] = np.sum(np.maximum(0, overlap))

    # find node and position for removed mutations
    # TODO: clean up, not sure if this is the best output format. it's not clear to me
    # that the mutation order will survive conversion from SINGER flat-file to tree sequence
    # (it probably won't) so the best return might simply be the surviving set.
    surviving = set()
    for j in range(masked_ts.num_mutations):
        pos = float(masked_ts.sites_position[masked_ts.mutations_site[j]])
        node = int(masked_ts.mutations_node[j])
        surviving.add((pos, node))

    removed_positions = []
    removed_nodes = []
    for j in range(ts.num_mutations):
        pos = float(ts.sites_position[ts.mutations_site[j]])
        node = int(ts.mutations_node[j])
        if (pos, node) not in surviving:
            removed_positions.append(pos)
            removed_nodes.append(node)
    removed_positions = np.array(removed_positions, dtype=np.float64)
    removed_nodes = np.array(removed_nodes, dtype=np.int32)

    return pruned_span, removed_positions, removed_nodes

