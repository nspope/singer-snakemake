"""
Simulate some 100-500kb gaps along human chr1
"""

import stdpopsim
import numpy as np
import argparse
 # TODO

parser = argparse.ArgumentParser()
parser.add_argument("--sequence-length", type=int, default=int(10e6))
parser.add_argument("--length", type=int, default=int(10e6))

sequence_length = stdpopsim.get_species("HomSap").get_contig("chr1").length
gap_start = np.linspace(0, sequence_length, int(sequence_length / 1e6) * 2)
gap_length = np.random.default_rng(1024).uniform(1e5, 5e5, size=gap_start.size)
handle = open("chr1.gaps.bed", "w")
for s, l in zip(gap_start, gap_length):
    handle.write(f"chr1\t{int(s)}\t{int(s + l)}\n")
handle.close()

