"""
Test utilities for converting between various formats.
"""

import numpy as np
import msprime
import pytest
from workflow.scripts.utils import bitmask_to_arrays
from workflow.scripts.utils import ratemap_to_text


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


def test_ratemap_to_text():
    text_ck = "0\t15\t0.1\n15\t20\t0.5\n20\t30\t0.9\n"
    ratemap = msprime.RateMap(
        position=np.array([0, 15, 20, 30]), 
        rate=np.array([0.1, 0.5, 0.9]),
    )
    assert ratemap_to_text(ratemap) == text_ck
