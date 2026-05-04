_docstring = """
Illustration that mapping to cM decreases variance in empirical pair
coalescence rates.

Simulate an ARG with a recombination rate that is "normal" in the flanks and
dips substantially in the middle 1/3 of the sequence. Calculate coalescence
rates in two coordinate systems: base pairs and recombination units. Do this
lots, while varying the severity of the "recombination dip".  Calculate the
error and bias in estimated N(t) relative to true N(t) [under the zigzag model]
for estimates from the two coordinate systems.
"""
import argparse
import numpy as np
import scipy.interpolate
import tskit
import stdpopsim
import matplotlib.pyplot as plt
import demes
from msprime import RateMap
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from workflow.scripts.validation.utils import transform_coordinates
from workflow.scripts.validation.utils import ratemap_to_hapmap


if __name__ == "__main__":

    parser = argparse.ArgumentParser(_docstring)
    parser.add_argument(
        "--output-prefix", type=str, 
        default="experiments/paircoal-weighting/paircoal-weighting", 
        help="Write figures/outputs with this prefix",
    )
    parser.add_argument(
        "--dip-factor-grid", type=float, nargs="+", 
        default=[1e-4, 1e-2, 1e-0],
        help="Grid of ratios to for the recombination dip",
    )
    parser.add_argument(
        "--num-replicates", type=int, 
        default=10, help="Number of replicates per dip",
    )
    parser.add_argument("--recombination-rate", type=float, default=1e-8)
    parser.add_argument("--time-grid-size", type=int, default=50)
    parser.add_argument("--sequence-length", type=float, default=1e7)
    parser.add_argument("--num-diploids", type=int, default=30)
    parser.add_argument("--max-cpus", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=1024)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    demographic_model = stdpopsim.get_species("HomSap").get_demographic_model("Zigzag_1S14").model
    recombination_maps = [
        RateMap(
            position=np.array([0, 1/3, 2/3, 1]) * args.sequence_length, 
            rate=np.array([1.0, dip_factor, 1.0]) * args.recombination_rate,
        ) for dip_factor in args.dip_factor_grid
    ]
    max_time = demographic_model.debug().mean_coalescence_time(lineages={"generic":2}) * 3
    time_grid = np.logspace(2, np.log10(max_time), args.time_grid_size + 1)

    # save model for further experimentation
    demes.dump(demographic_model.to_demes(), f"{args.output_prefix}.demes.yaml")
    for i, ratemap in enumerate(recombination_maps):
        with open(f"{args.output_prefix}.{i}.hapmap", "w") as handle:
            handle.write(ratemap_to_hapmap(ratemap, "1"))

    # interpolate true rates with a spline, then average over grid intervals
    fine_time_grid = np.logspace(np.log10(time_grid.min()), np.log10(time_grid.max()), 10000)
    fine_coal_rates = 0.5 / demographic_model.debug().population_size_trajectory(fine_time_grid)
    true_rate_spline = scipy.interpolate.make_interp_spline(fine_time_grid, fine_coal_rates, k=3)
    true_rate_average = np.array([
        true_rate_spline.integrate(a, b) / (b-a) 
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

    # calculate rates over simulations in both coordinate systems
    def calculate_rates(
        seed: int, 
        ratemap: RateMap,
    ) -> (np.ndarray, np.ndarray):
        import msprime
        ts_generator = msprime.sim_ancestry(
            samples={"generic": args.num_diploids},
            recombination_rate=ratemap,
            demography=demographic_model,
            random_seed=seed,
            num_replicates=args.num_replicates,
        )
        rates_bp = []
        rates_cm = []
        for ts_bp in ts_generator:
            ts_cm = transform_coordinates(ts_bp, ratemap=ratemap)
            rates_bp.append(
                ts_bp.pair_coalescence_rates(
                    time_windows=np.append(np.append(0, time_grid), np.inf)
                )[1:-1]
            )
            rates_cm.append(
                ts_cm.pair_coalescence_rates(
                    time_windows=np.append(np.append(0, time_grid), np.inf)
                )[1:-1]
            )
        return np.stack(rates_bp), np.stack(rates_cm)

    seed_array = rng.integers(1, 2 ** 32 - 1, size=len(recombination_maps))
    rates_bp_grid, rates_cm_grid = zip(*[
        calculate_rates(seed, recmap)
        for seed, recmap in zip(seed_array, recombination_maps)
    ])
            
    # plot empirical rates
    cols = 2
    rows = len(recombination_maps)
    colors = ["firebrick", "dodgerblue"]
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 3), 
        sharex=True, sharey=True, constrained_layout=True,
    )
    for i, rates_pair in enumerate(zip(rates_bp_grid, rates_cm_grid)):
        for j, rates in enumerate(rates_pair):
            for x in rates:
                axs[i, j].step(
                    time_grid[:-1], x, 
                    where="post", color=colors[j], alpha=0.5,
                )
            axs[i, j].step(
                time_grid[:-1], true_rate_average, 
                where="post", color="black", linewidth=1.5,
            )
            axs[i, j].set_xscale("log")
        axs[i, 0].text(
            0.05, 0.95, 
            f"Recomb dip factor: {args.dip_factor_grid[i]}", 
            transform=axs[i, 0].transAxes, 
            ha="left", va="top",
        )
        if i == 0: axs[i, 0].set_title("base pairs")
        if i == 0: axs[i, 1].set_title("recomb units")
    fig.supylabel("1 / (2 N(t))")
    fig.supxlabel("Generations ago (t)")
    plt.savefig(f"{args.output_prefix}.sim_rates.png")
    plt.clf()

    # plot mean, std
    cols = 2
    rows = 2
    dims = len(recombination_maps)
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 3), 
        sharex=True, sharey="row", constrained_layout=True,
    )
    midpoints = (time_grid[:-1] + time_grid[1:]) / 2
    cmap = plt.get_cmap("Paired")
    for i, rates_pair in enumerate(zip(rates_bp_grid, rates_cm_grid)):
        for j, rates in enumerate(rates_pair):
            bias = rates.mean(axis=0) - true_rate_average
            rmse = np.sqrt(np.mean(np.power(rates - true_rate_average[np.newaxis], 2), axis=0))
            color = cmap(i / (dims - 1))
            label = f"{args.dip_factor_grid[i]}" 
            axs[0, j].plot(midpoints, bias, "-o", color=color, label=label, markersize=2)
            axs[1, j].plot(midpoints, rmse, "-o", color=color, label=label, markersize=2)
    axs[0, 0].axhline(y=0, color="black", linestyle="dashed")
    axs[0, 1].axhline(y=0, color="black", linestyle="dashed")
    axs[0, 0].set_ylabel("Bias", size=10)
    axs[1, 0].set_ylabel("RMSE", size=10)
    axs[1, 1].legend(title="Recomb dip factor")
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_title("base pairs")
    axs[0, 1].set_title("recomb units")
    fig.supxlabel("Generations ago (t)")
    plt.savefig(f"{args.output_prefix}.errors.png")
    plt.clf()

