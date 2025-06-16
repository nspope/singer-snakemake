import tskit
import tszip
import pickle
import numpy as np
import os
import argparse

from collections import defaultdict
import matplotlib.pyplot as plt


docstring = \
"""
Compare the timescales of true and inferred ARGs, in terms of mutation ages
(across frequencies). 
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(docstring)
    parser.add_argument(
        "--true-arg", type=str, required=True,
        help="Path to true ARG (tree sequence), e.g. from simulation",
    )
    parser.add_argument(
        "--inferred-arg", type=str, required=True,
        help="Path to inferred ARG (tree sequence), e.g. a SINGER replicate",
    )
    parser.add_argument(
        "--inaccessible-ratemap", type=str, required=True,
        help="Path to msprime.RateMap with the proportion of inaccessible sequence per chunk",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to directory to place plots (will be created if nonexistant)",
    )
    parser.add_argument(
        "--coalrate-epochs", type=int, default=25,
        help="Number of (logarithmic) time intervals for pair coalescence pdf",
    )
    parser.add_argument(
        "--plot-by-chunk", action="store_true",
        help="Rather than plot global statistics, plot by chunk",
    )
    parser.add_argument(
        "--no-position-adjust", action="store_true",
        help="By default positions are assumed to be 0-based in simulated tree sequence, "
        "and 1-based in the inferred tree sequence. Use this flag to disable this assumption.",
    )
    parser.add_argument(
        "--random-seed", type=int, default=1024,
        help="Random seed for sampling mutation position on a branch",
    )
    args = parser.parse_args()
    
    inaccessible = pickle.load(open(args.inaccessible_ratemap, "rb"))
    true_ts = tszip.load(args.true_arg)
    infr_ts = tszip.load(args.inferred_arg)
    assert true_ts.num_samples == infr_ts.num_samples
    assert true_ts.sequence_length == infr_ts.sequence_length
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    # subset mutations down to those present in the inferred ts, that
    # are biallelic and have not been flipped by SINGER
    biallelic = np.bincount(infr_ts.mutations_site, minlength=infr_ts.num_sites) == 1
    id_position = infr_ts.sites_position[biallelic]
    position_id = {p: i for i, p in enumerate(id_position)}

    mutation_freq_true = np.full(len(position_id), tskit.NULL)
    mutation_time_true = np.full(len(position_id), np.nan)
    for t in true_ts.trees():
        for m in t.mutations():
            p = true_ts.sites_position[m.site] + 1
            if p in position_id:
                assert m.edge != tskit.NULL
                i = position_id[p]
                mutation_freq_true[i] = t.num_samples(m.node)
                # TODO: use true time, not midpoint?
                mutation_time_true[i] = (t.time(t.parent(m.node)) + t.time(m.node)) / 2
    assert np.all(mutation_freq_true >= 0)

    mutation_freq_infr = np.full(len(position_id), tskit.NULL)
    mutation_time_infr = np.full(len(position_id), np.nan)
    for t in infr_ts.trees():
        for m in t.mutations():
            p = infr_ts.sites_position[m.site]
            if p in position_id:
                assert m.edge != tskit.NULL
                i = position_id[p]
                mutation_freq_infr[i] = t.num_samples(m.node)
                # TODO: sample uniformly on branch, not midpoint?
                mutation_time_infr[i] = (t.time(t.parent(m.node)) + t.time(m.node)) / 2
    assert np.all(mutation_freq_infr >= 0)

    mismatch = mutation_freq_true != mutation_freq_infr
    #assert np.allclose(mutation_freq_true, mutation_freq_infr)
    print(f"Omitting {sum(mismatch)} of {mismatch.size} mutations that are mapped to the wrong clade")
    mutation_freq_true = mutation_freq_true[~mismatch]
    mutation_time_true = mutation_time_true[~mismatch]
    mutation_freq_infr = mutation_freq_infr[~mismatch]
    mutation_time_infr = mutation_time_infr[~mismatch]
    
    freq_counts = np.bincount(mutation_freq_true, minlength=true_ts.num_samples)
    true_mean = np.bincount(mutation_freq_true, weights=mutation_time_true, minlength=true_ts.num_samples)
    true_mean[freq_counts > 0] /= freq_counts[freq_counts > 0]
    infr_mean = np.bincount(mutation_freq_infr, weights=mutation_time_infr, minlength=infr_ts.num_samples)
    infr_mean[freq_counts > 0] /= freq_counts[freq_counts > 0]

    rows = 1
    cols = 1
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), constrained_layout=True)
    axs.plot(
        np.arange(true_ts.num_samples)[freq_counts > 0], 
        true_mean[freq_counts > 0], "o", 
        color="black", markersize=4, label="true",
    )
    axs.plot(
        np.arange(true_ts.num_samples)[freq_counts > 0], 
        infr_mean[freq_counts > 0], "o", 
        color="red", markersize=4, label="infr",
    )
    axs.legend()
    axs.set_ylabel("Mean mutation age")
    axs.set_xlabel("Mutation frequency")
    axs.set_yscale("log")
    plt.savefig(f"{args.output_dir}/simulated-vs-inferred-mutation-ages.png")


