"""
Plot pair coalescence time distribution and pair coalescence rates between
true and inferred ARGs
"""

import numpy as np
import matplotlib
import itertools
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 300

time_grid = snakemake.params.time_grid
reference_density = np.load(next(iter(snakemake.input.infr_pair_density)))
num_replicates = len(snakemake.input.infr_pair_density)

# average over simulations
infr_pair_density = np.zeros_like(reference_density)
for file in snakemake.input.infr_pair_density:
    infr_pair_density += np.load(file) / num_replicates

true_pair_density = np.zeros_like(reference_density)
for file in snakemake.input.true_pair_density:
    true_pair_density += np.load(file) / num_replicates

infr_pair_rates = np.zeros_like(reference_density)
for file in snakemake.input.infr_pair_rates:
    infr_pair_rates += np.load(file) / num_replicates

true_pair_rates = np.zeros_like(reference_density)
for file in snakemake.input.true_pair_rates:
    true_pair_rates += np.load(file) / num_replicates

#infr_pair_survival = np.concatenate([
#    np.ones((*reference_density.shape[:2], 1)),
#    1 - np.cumsum(infr_pair_density, axis=-1),
#])
#
#true_pair_survival = np.concatenate([
#    np.ones((*reference_density.shape[:2], 1)),
#    1 - np.cumsum(true_pair_density, axis=-1),
#])

# plot pair coalescence density
rows, cols = reference_density.shape[:2]
fig, axs = plt.subplots(
    rows, cols, 
    figsize=(2.25 * cols, 2 * rows) if cols > 1 else (4, 3.25), 
    sharex=True, sharey=True, squeeze=False,
)
population_pairs = itertools.product(range(rows), range(cols))
for (i, j) in population_pairs:
    if i <= j:
        label = f"({i}, {j})"
        axs[i, j].step(time_grid[:-2], true_pair_density[i, j][1:-1], color="black", label="true")
        axs[i, j].step(time_grid[:-2], infr_pair_density[i, j][1:-1], color="firebrick", label="estimated", alpha=0.8)
        axs[i, j].set_xscale("log")
        if rows > 1:
            axs[i, j].text(0.05, 0.95, label, ha="left", va="top", transform=axs[i, j].transAxes, size=10)
        if i == j:
            axs[i, j].tick_params(axis="both", labelbottom=True, labelleft=True)
    else:
        axs[i, j].set_visible(False)
fig.supxlabel("Generations in the past")
fig.supylabel("Proportion coalescing pairs")
if cols > 1:
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(), 
        loc="lower left",
        bbox_to_anchor=(0.15, 0.15),
    )
else:
    axs[0, 0].legend()
fig.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.savefig(snakemake.output.pair_den_plot)
plt.clf()

# plot pair coalescence rates
rows, cols = reference_density.shape[:2]
fig, axs = plt.subplots(
    rows, cols, 
    figsize=(2.25 * cols, 2 * rows) if cols > 1 else (4, 3.25), 
    sharex=True, sharey=True, squeeze=False,
)
population_pairs = itertools.product(range(rows), range(cols))
for (i, j) in population_pairs:
    if i <= j:
        label = f"({i}, {j})"
        axs[i, j].step(time_grid[:-2], true_pair_rates[i, j][1:-1], color="black", label="true")
        axs[i, j].step(time_grid[:-2], infr_pair_rates[i, j][1:-1], color="firebrick", label="estimated", alpha=0.8)
        axs[i, j].set_xscale("log")
        if snakemake.params.log_coalescence_rates: 
            axs[i, j].set_yscale("log")
        if rows > 1:
            axs[i, j].text(0.95, 0.95, label, ha="right", va="top", transform=axs[i, j].transAxes, size=10)
        if i == j:
            axs[i, j].tick_params(axis="both", labelbottom=True, labelleft=True)
    else:
        axs[i, j].set_visible(False)
fig.supxlabel("Generations in the past")
fig.supylabel("Pair coalescence rate")
if cols > 1:
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(), 
        loc="lower left",
        bbox_to_anchor=(0.15, 0.15),
    )
else:
    axs[0, 0].legend()
fig.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.savefig(snakemake.output.pair_rat_plot)
plt.clf()
