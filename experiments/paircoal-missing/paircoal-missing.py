# /// script
# dependencies = [
#   "matplotlib",
#   "msprime",
#   "numpy",
#   "tskit @ git+https://github.com/nspope/tskit.git@nsp-paircoal-partial-missing#subdirectory=python",
#   "stdpopsim",
#   "scipy",
#   "demes",
# ]
# ///

# build options for tskit:
# "tskit @ file:///home/natep/projects/tskit/tskit/python"
# "tskit @ git+https://github.com/nspope/tskit.git@nsp-paircoal-partial-missing#subdirectory=python"
# "tskit==1.0.2"

_docstring = """
Test coalescence rate calculation when there are variable numbers of samples
across the chromosome.

Simulate an ARG, apply a highly dissected sample mask, calculate coalescence
rates on original and dissected ARG.
"""

import argparse
import numpy as np
import scipy.interpolate
import tskit
import stdpopsim
import matplotlib.pyplot as plt
import msprime
import demes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from workflow.scripts.utils import parse_sample_bedmask
from workflow.scripts.utils import remove_partial_ancestry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(_docstring)
    parser.add_argument(
        "--output-prefix", type=str,
        default="experiments/paircoal-missing/paircoal-missing",
        help="Write figures/outputs with this prefix",
    )
    parser.add_argument(
        "--sample-mask-bed", type=str,
        default="experiments/paircoal-missing/sample.mask.bed",
        help="Bed file with the sample mask, fourth column is sample name",
    )
    parser.add_argument("--num-replicates", type=int, default=10)
    parser.add_argument("--recombination-rate", type=float, default=1e-8)
    parser.add_argument("--time-grid-size", type=int, default=50)
    parser.add_argument("--sequence-length", type=float, default=1e7)
    parser.add_argument("--num-diploids", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=1024)
    args = parser.parse_args()

    demographic_model = stdpopsim.get_species("HomSap").get_demographic_model("Zigzag_1S14").model

    # parse per-sample bedmask, then expand from individual to haploid node ids
    sample_map = {f"Sample{i+1:03d}": i for i in range(args.num_diploids)}
    intervals_by_individual = parse_sample_bedmask(args.sample_mask_bed, sample_map)
    sample_mask = {}
    for ind_id, intervals in intervals_by_individual.items():
        sample_mask[2 * ind_id] = intervals
        sample_mask[2 * ind_id + 1] = intervals

    max_time = demographic_model.debug().mean_coalescence_time(lineages={"generic": 2}) * 3
    time_grid = np.logspace(2, np.log10(max_time), args.time_grid_size + 1)
    time_windows = np.append(np.append(0, time_grid), np.inf)

    # save model for further experimentation
    demes.dump(demographic_model.to_demes(), f"{args.output_prefix}.demes.yaml")

    # interpolate true rates with a spline, then average over grid intervals
    fine_time_grid = np.logspace(np.log10(time_grid.min()), np.log10(time_grid.max()), 10000)
    fine_coal_rates = 0.5 / demographic_model.debug().population_size_trajectory(fine_time_grid)
    true_rate_spline = scipy.interpolate.make_interp_spline(fine_time_grid, fine_coal_rates, k=3)
    true_rate_average = np.array([
        true_rate_spline.integrate(a, b) / (b - a)
        for a, b in zip(time_grid[:-1], time_grid[1:])
    ]).flatten()

    # plot averages as a sanity check
    fig, axs = plt.subplots(1, figsize=(5, 3.5), constrained_layout=True)
    axs.plot(fine_time_grid, fine_coal_rates, "-", color="red", label="true-smooth")
    axs.step(time_grid[:-1], true_rate_average, where="post", color="black", label="true-average")
    axs.set_xscale("log")
    axs.set_ylabel("1 / (2 N(t))")
    axs.set_xlabel("Generations ago (t)")
    axs.legend()
    plt.savefig(f"{args.output_prefix}.true_rates.png")
    plt.clf()

    # simulate replicates, drop ancestry, compute rates with and without mask
    ts_generator = msprime.sim_ancestry(
        samples={"generic": args.num_diploids},
        recombination_rate=args.recombination_rate,
        sequence_length=args.sequence_length,
        demography=demographic_model,
        random_seed=args.random_seed,
        num_replicates=args.num_replicates,
    )
    rates_full = []
    rates_drop = []
    for ts_full in ts_generator:
        ts_drop = remove_partial_ancestry(ts_full, sample_mask)
        rates_full.append(ts_full.pair_coalescence_rates(time_windows=time_windows)[1:-1])
        rates_drop.append(ts_drop.pair_coalescence_rates(time_windows=time_windows)[1:-1])
    rates_full = np.stack(rates_full)
    rates_drop = np.stack(rates_drop)

    # plot empirical rates per replicate alongside truth
    cols = 2
    colors = ["firebrick", "dodgerblue"]
    titles = ["full", "drop"]
    fig, axs = plt.subplots(
        1, cols, figsize=(cols * 4, 3),
        sharex=True, sharey=True, constrained_layout=True,
    )
    for j, rates in enumerate([rates_full, rates_drop]):
        for x in rates:
            axs[j].step(time_grid[:-1], x, where="post", color=colors[j], alpha=0.5)
        axs[j].step(
            time_grid[:-1], true_rate_average,
            where="post", color="black", linewidth=1.5,
        )
        axs[j].set_xscale("log")
        axs[j].set_title(titles[j])
    fig.supylabel("1 / (2 N(t))")
    fig.supxlabel("Generations ago (t)")
    plt.savefig(f"{args.output_prefix}.sim_rates.png")
    plt.clf()

    # plot bias and RMSE
    rows = 2
    fig, axs = plt.subplots(
        rows, 1, figsize=(5, rows * 3),
        sharex=True, constrained_layout=True,
    )
    midpoints = (time_grid[:-1] + time_grid[1:]) / 2
    for rates, color, label in zip([rates_full, rates_drop], colors, titles):
        bias = rates.mean(axis=0) - true_rate_average
        rmse = np.sqrt(np.mean(np.power(rates - true_rate_average[np.newaxis], 2), axis=0))
        axs[0].plot(midpoints, bias, "-o", color=color, label=label, markersize=2)
        axs[1].plot(midpoints, rmse, "-o", color=color, label=label, markersize=2)
    axs[0].axhline(y=0, color="black", linestyle="dashed")
    axs[0].set_ylabel("Bias")
    axs[1].set_ylabel("RMSE")
    axs[0].set_xscale("log")
    axs[1].legend()
    fig.supxlabel("Generations ago (t)")
    plt.savefig(f"{args.output_prefix}.errors.png")
    plt.clf()
