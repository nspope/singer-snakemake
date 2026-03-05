"""
Test utilities for converting between various formats.
"""

import numpy as np
import msprime
import pytest
from workflow.scripts.utils import bitmask_to_arrays
from workflow.scripts.utils import ratemap_to_text
from workflow.scripts.utils import ratemap_product
from workflow.scripts.utils import compactify_run_length_encoding


def test_bitmask_to_arrays():
    breakpoints_ck = np.array([0, 3, 5, 9, 30, 50, 100, 400, 500, 600, 1000]).astype(int)
    values_ck = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    bed_intervals = np.array([[0, 3], [9, 30], [400, 500]])
    bitmask = np.full(1000, False)
    for (a, b) in bed_intervals: bitmask[a:b] = True
    extra_breakpoints = np.array([0, 5, 50, 100, 400, 600, 1000])
    values, breakpoints = bitmask_to_arrays(bitmask, insert_breakpoints=extra_breakpoints)
    assert values.size == breakpoints.size - 1
    np.testing.assert_allclose(breakpoints_ck, breakpoints)
    np.testing.assert_allclose(values_ck, values)


@pytest.mark.skip("broken until precision is set")
def test_ratemap_to_text():
    text_ck = "0\t15\t0.1\n15\t20\t0.5\n20\t30\t0.9\n"
    ratemap = msprime.RateMap(
        position=np.array([0, 15, 20, 30]), 
        rate=np.array([0.1, 0.5, 0.9]),
    )
    assert ratemap_to_text(ratemap) == text_ck


def test_ratemap_product():
    rng = np.random.default_rng(10)
    grid = np.linspace(0, 1, 1000)[:-1]
    n, m = 40, 20
    position = np.unique(np.concatenate([[0, 1], rng.uniform(size=n - 2)]))
    rate = rng.uniform(size=n - 1)
    ratemap = msprime.RateMap(position=position, rate=rate)
    other_position = np.unique(np.concatenate([[0, 1], rng.uniform(size=m - 2)]))
    other_rate = rng.uniform(size=m - 1)
    other = msprime.RateMap(position=other_position, rate=other_rate)
    ratemap_prod = ratemap_product(ratemap, other)
    np.testing.assert_allclose(
        ratemap_prod.get_rate(grid), 
        ratemap.get_rate(grid) * other.get_rate(grid),
    )


def test_ratemap_square():
    rng = np.random.default_rng(10)
    grid = np.linspace(0, 1, 1000)[:-1]
    n, m = 40, 20
    position = np.unique(np.concatenate([[0, 1], rng.uniform(size=n - 2)]))
    rate = rng.uniform(size=n - 1)
    ratemap = msprime.RateMap(position=position, rate=rate)
    ratemap_sq = ratemap_product(ratemap, ratemap)
    np.testing.assert_allclose(ratemap_sq.get_rate(grid), ratemap.get_rate(grid) ** 2)


def test_ratemap_nan():
    rng = np.random.default_rng(10)
    grid = np.linspace(0, 1, 1000)[:-1]
    n, m = 40, 20
    position = np.unique(np.concatenate([[0, 1], rng.uniform(size=n - 2)]))
    rate = rng.uniform(size=n - 1)
    ratemap = msprime.RateMap(position=position, rate=rate)
    other = msprime.RateMap(
        position=np.linspace(0, 1, 5),
        rate=np.array([np.nan, np.nan, 1.0, np.nan])
    )
    ratemap_prod = ratemap_product(ratemap, other)
    np.testing.assert_allclose(
        ratemap_prod.get_rate(grid), 
        ratemap.get_rate(grid) * other.get_rate(grid),
    )


def test_compactify_run_length_encoding():
    rng = np.random.default_rng(10)
    n, m = 1000, 30
    position = np.unique(np.concatenate([[0, 1], rng.uniform(size=n - 2)]))
    rate = rng.uniform(size=n - 1)
    ratemap = msprime.RateMap(position=position, rate=rate)
    other_position = np.unique(np.concatenate([[0, 1], rng.uniform(size=m - 2)]))
    other_rate = rng.uniform(size=m - 1) > 0.5
    other = msprime.RateMap(position=other_position, rate=other_rate)
    ratemap_prod = ratemap_product(ratemap, other)
    breaks, values = ratemap_prod.position, ratemap_prod.rate
    breaks_ck = [breaks[0]]
    values_ck = [values[0]]
    for brk, val in zip(breaks[1:-1], values[1:]):
        if val != values_ck[-1]:
            breaks_ck.append(brk)
            values_ck.append(val)
    breaks_ck.append(breaks[-1])
    breaks_ck, values_ck = np.array(breaks_ck), np.array(values_ck)
    assert breaks_ck.size < breaks.size
    assert values_ck.size < values.size
    breaks, values = compactify_run_length_encoding(breaks, values)
    np.testing.assert_allclose(breaks, breaks_ck)
    np.testing.assert_allclose(values, values_ck)
