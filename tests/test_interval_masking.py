"""
Test utilities for masking ancestry
"""

import numpy as np
import msprime
import tskit
import pytest

from workflow.scripts.utils import merge_intervals
from workflow.scripts.utils import clip_and_shift_intervals
from workflow.scripts.utils import interval_coverage
from workflow.scripts.utils import remove_partial_ancestry
from workflow.scripts.utils import adjust_edge_spans_for_partial_ancestry


def _2d(*x):
    return np.array(x, dtype=float).reshape(-1, 2)


def test_merge_intervals():
    merge = lambda *x: merge_intervals(_2d(x)).tolist()
    assert merge((10, 20), (30, 40)) == [[10, 20], [30, 40]]  # non-overlapping and sorted
    assert merge((10, 30), (20, 40)) == [[10, 40]]  # overlapping
    assert merge((10, 20), (20, 30)) == [[10, 30]]  # bookended
    assert merge((30, 40), (10, 20)) == [[10, 20], [30, 40]]  # unsorted
    assert merge((30, 50), (10, 40)) == [[10, 50]]  # unsorted and overlapping
    assert merge((10, 50), (20, 30)) == [[10, 50]]  # nested
    assert merge((10, 20)) == [[10, 20]]  # single interval
    assert merge_intervals(np.empty((0, 2))).shape == (0, 2)  # empty
    assert merge((1, 5), (3, 8), (7, 12), (11, 15)) == [[1, 15]]  # all overlapping
    assert merge((0, 10), (10, 20), (20, 30)) == [[0, 30]]  # bookended chain
    assert merge((0, 10), (5, 15), (30, 40), (35, 50)) == [[0, 15], [30, 50]]  # merged and distinct


def test_clip_and_shift_intervals():
    f = clip_and_shift_intervals
    # basic clipping and shifting
    intervals = f(_2d((10, 50), (60, 90)), start=20, end=80)
    assert np.array_equal(intervals, _2d((0, 30), (40, 60)))
    # without shift
    intervals = f(_2d((10, 50), (60, 90)), start=20, end=80, shift=False)
    assert np.array_equal(intervals, _2d((20, 50), (60, 80)))
    # interval entirely outside
    intervals = f(_2d((10, 20)), start=30, end=50)
    assert intervals.shape == (0, 2)
    # interval touching boundary
    intervals = f(_2d((50, 70)), start=30, end=50)
    assert intervals.shape == (0, 2)
    # interval straddling start
    intervals = f(_2d((10, 40)), start=20, end=100)
    assert np.array_equal(intervals, _2d((0, 20)))
    # interval straddling end
    intervals = f(_2d((60, 90)), start=0, end=80)
    assert np.array_equal(intervals, _2d((60, 80)))
    # all intervals inside so shift only
    intervals = f(_2d((20, 30), (40, 50)), start=10, end=60)
    assert np.array_equal(intervals, _2d((10, 20), (30, 40)))
    # default arguments leave unchanged
    intervals = f(_2d((10, 20)))
    assert np.array_equal(intervals, _2d((10, 20)))
    # one partially overlaps, one entirely outside
    intervals = f(_2d((10, 40), (60, 90)), start=30, end=70)
    assert np.array_equal(intervals, _2d((0, 10), (30, 40)))
    # empty input
    intervals = f(np.empty((0, 2)), start=10, end=50)
    assert intervals.shape == (0, 2)


def test_interval_coverage():
    def _check(x, expected_intervals, expected_coverage):
        intervals, coverage = x
        expected_intervals = np.array(expected_intervals, dtype=float) \
            if expected_intervals else np.empty((0, 2))
        expected_coverage = np.array(expected_coverage, dtype=int)
        assert np.array_equal(intervals, expected_intervals)
        assert np.array_equal(coverage, expected_coverage)
    def _filter(x, minimum_coverage):
        intervals, coverage = x
        return intervals[coverage >= minimum_coverage]
    f = interval_coverage
    # book-ended
    x = f({0: _2d((10, 30)), 1: _2d((30, 50))})
    _check(x, [(10, 30), (30, 50)], [1, 1])
    s, e = _filter(x, 2).T
    assert s.size == 0
    # identical intervals across samples
    x = f({0: _2d((10, 40)), 1: _2d((10, 40)), 2: _2d((10, 40))})
    _check(x, [(10, 40)], [3])
    s, e = _filter(x, 2).T
    assert np.array_equal(np.column_stack([s, e]), [[10, 40]])
    s, e = _filter(x, 4).T
    assert s.size == 0
    # partial overlap
    x = f({0: _2d((10, 40)), 1: _2d((30, 60))})
    _check(x, [(10, 30), (30, 40), (40, 60)], [1, 2, 1])
    s, e = _filter(x, 2).T
    assert np.array_equal(np.column_stack([s, e]), [[30, 40]])
    # same-sample book-ended
    x = f({0: _2d((10, 20), (20, 30))})
    _check(x, [(10, 20), (20, 30)], [1, 1])
    # no overlap at all
    x = f({0: _2d((10, 20)), 1: _2d((30, 40))})
    _check(x, [(10, 20), (20, 30), (30, 40)], [1, 0, 1])
    s, e = _filter(x, 2).T
    assert s.size == 0
    # single sample, single interval
    x = f({0: _2d((10, 20))})
    _check(x, [(10, 20)], [1])
    # many samples, same region
    d = {i: _2d((0, 100)) for i in range(5)}
    x = f(d)
    _check(x, [(0, 100)], [5])
    for cov in range(1, 6):
        s, e = _filter(x, cov).T
        assert np.array_equal(np.column_stack([s, e]), [[0, 100]])
    s, e = _filter(x, 6).T
    assert s.size == 0
    # nested intervals
    x = f({0: _2d((10, 50)), 1: _2d((20, 30))})
    _check(x, [(10, 20), (20, 30), (30, 50)], [1, 2, 1])
    s, e = _filter(x, 2).T
    assert np.array_equal(np.column_stack([s, e]), [[20, 30]])
    # three-way book-end
    x = f({0: _2d((10, 30)), 1: _2d((30, 50)), 2: _2d((30, 70))})
    _check(x, [(10, 30), (30, 50), (50, 70)], [1, 2, 1])
    s, e = _filter(x, 2).T
    assert np.array_equal(np.column_stack([s, e]), [[30, 50]])


@pytest.mark.parametrize("random_seed", [1, 1024, 123123])
@pytest.mark.parametrize("num_intervals", [100])
@pytest.mark.parametrize("missing_data", [True, False])
def test_remove_partial_ancestry(random_seed, num_intervals, missing_data):
    rng = np.random.default_rng(random_seed)
    orig_ts = msprime.sim_ancestry(10, sequence_length=1e5, population_size=1e4, recombination_rate=1e-8, random_seed=random_seed)
    orig_ts = msprime.sim_mutations(orig_ts, rate=1e-7, random_seed=random_seed)
    # simulate missingness
    samples_with_missing = rng.choice(list(orig_ts.samples()), size=10, replace=False)
    intervals_by_sample = {}
    for i in samples_with_missing:
        intervals_by_sample[i] = np.sort(rng.uniform(0, orig_ts.sequence_length, size=num_intervals * 2)).reshape(-1, 2)
    if missing_data:
        missing_entirely = np.sort(rng.uniform(0, orig_ts.sequence_length, size=num_intervals * 2)).reshape(-1, 2)
        orig_ts = orig_ts.delete_intervals(missing_entirely)
    # drop intervals per sample
    ts = remove_partial_ancestry(orig_ts, intervals_by_sample)
    # test that ancestry has been removed inside intervals and not outside intervals
    for i in samples_with_missing:
        tree = ts.first()
        last_right = -np.inf
        for (left, right) in intervals_by_sample[i]:
            if left != last_right and np.isfinite(last_right):
                tree.seek(last_right)
                if tree.num_edges:
                    assert tree.edge(i) != tskit.NULL
            tree.seek(left)
            if tree.num_edges:
                assert tree.edge(i) == tskit.NULL
            last_right = right
    # test that there are no dangling nodes
    for tree in ts.trees():
        for node in tree.nodes():
            if tree.num_children(node) == 0:
                assert tree.is_sample(node)
    # test that there are no mutations on removed ancestry
    # (this check will fail if there are mutations above the root)
    for tree in ts.trees():
        for mut in tree.mutations():
            assert mut.edge != tskit.NULL


@pytest.mark.parametrize("random_seed", [1, 1024, 123123])
@pytest.mark.parametrize("num_intervals", [100])
@pytest.mark.parametrize("missing_data", [True, False])
def test_adjust_edge_spans_for_missing_ancestry(random_seed, num_intervals, missing_data):
    # simulate tree sequence and missing intervals
    rng = np.random.default_rng(random_seed)
    ts = msprime.sim_ancestry(
        10, sequence_length=1e5, population_size=1e4,
        recombination_rate=1e-8, random_seed=random_seed,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=random_seed)
    samples_with_missing = rng.choice(
        list(ts.samples()), size=10, replace=False,
    )
    intervals_by_sample = {}
    for i in samples_with_missing:
        intervals_by_sample[i] = np.sort(
            rng.uniform(0, ts.sequence_length, size=num_intervals * 2)
        ).reshape(-1, 2)
    if missing_data:
        missing_entirely = np.sort(
            rng.uniform(0, ts.sequence_length, size=num_intervals * 2)
        ).reshape(-1, 2)
        ts = ts.delete_intervals(missing_entirely)
    marked_samples = set(samples_with_missing)
    # get corrected spans, etc.
    corrected, rem_pos, rem_node = adjust_edge_spans_for_partial_ancestry(ts, intervals_by_sample)
    original_spans = ts.edges_right - ts.edges_left
    assert np.all(corrected >= 0.0)
    assert np.all(corrected <= original_spans)
    # total corrected span <= total original span
    assert corrected.sum() <= original_spans.sum() + 1e-6
    # edges whose child is NOT a marked sample and whose subtree has at least
    # one unmarked sample should be untouched, so edges where child is an
    # unmarked sample should have full span
    unmarked_samples = set(ts.samples().tolist()) - marked_samples
    for i in range(ts.num_edges):
        child = ts.edges_child[i]
        if child in unmarked_samples:
            assert np.isclose(corrected[i], original_spans[i])
    # edges where child IS a marked sample and the edge is entirely inside
    # the sample's mask intervals should have corrected_span == 0
    for sample, intervals in intervals_by_sample.items():
        if intervals.size == 0:
            continue
        il, ir = intervals.T
        sample_edges = np.flatnonzero(ts.edges_child == sample)
        for ei in sample_edges:
            el = ts.edges_left[ei]
            er = ts.edges_right[ei]
            # check if [el, er) is fully contained in any single interval
            contained = np.any((il <= el) & (er <= ir))
            if contained: assert np.isclose(corrected[ei], 0.0)
    # removed mutations should be at positions inside some sample's mask
    # (either a sample-node mutation directly, or an internal-node mutation
    # whose subtree was fully masked)
    for pos, node in zip(rem_pos, rem_node):
        if node in marked_samples:
            intervals = intervals_by_sample[node]
            il, ir = intervals.T
            inside = np.any((il <= pos) & (pos < ir))
            assert inside
    # mutations at unmarked samples should never be removed
    for node in rem_node: assert node not in unmarked_samples


def test_adjust_edge_spans_for_partial_ancestry_with_empty_intervals():
    ts = msprime.sim_ancestry(
        5, sequence_length=1e4, population_size=1e4,
        recombination_rate=1e-8, random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=1)
    # empty intervals should leave everything unchanged
    corrected, rem_pos, rem_node = adjust_edge_spans_for_partial_ancestry(ts, {})
    original_spans = ts.edges_right - ts.edges_left
    assert np.allclose(corrected, original_spans)
    assert rem_pos.size == 0
    assert rem_node.size == 0


def test_adjust_edge_spans_for_partial_ancestry_with_full_coverage():
    ts = msprime.sim_ancestry(
        5, sequence_length=1e4, population_size=1e4,
        recombination_rate=1e-8, random_seed=2,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=2)
    # masking a sample over the entire sequence should zero its edges
    sample = ts.samples()[0]
    full_interval = np.array([[0.0, ts.sequence_length]])
    corrected, rem_pos, rem_node = adjust_edge_spans_for_partial_ancestry(ts, {sample: full_interval})
    for i in range(ts.num_edges):
        if ts.edges_child[i] == sample: 
            assert np.isclose(corrected[i], 0.0)

    # all mutations at this sample should be removed
    for j in range(ts.num_mutations):
        if ts.mutations_node[j] == sample:
            pos = ts.sites_position[ts.mutations_site[j]]
            assert pos in rem_pos, (
                f"mutation at fully-masked sample {sample} pos={pos} should be removed"
            )


