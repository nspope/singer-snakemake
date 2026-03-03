"""
Utility functions for converting between formats and manipulating tree sequences.

Part of https://github.com/nspope/singer-snakemake.
"""

import numpy as np
import msprime
import typing
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
    # TODO: vectorise this
    assert breaks.size == values.size + 1
    new_breaks = [breaks[0]]
    new_values = [values[0]]
    for brk, val in zip(breaks[1:-1], values[1:]):
        if val != new_values[-1]:
            new_breaks.append(brk)
            new_values.append(val)
    new_breaks.append(breaks[-1])
    return np.array(new_breaks), np.array(new_values)


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


