"""
Plot cumulative relatedness over time, that is the average number of shared mutations
younger than time "t" between haplotypes of a given pair of individuals. 

A small (but equally spaced) subset of individuals are plotted.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

matplotlib.rcParams["figure.dpi"] = 300

time_grid = np.append(0, snakemake.params.time_grid[:-1])
reference_rel = np.load(next(iter(snakemake.input.true_site_relatedness)))[..., 0]
num_replicates = len(snakemake.input.true_site_relatedness)
num_individuals = reference_rel.shape[0]
max_individuals = snakemake.params.max_individuals

# average over simulations
true_rel = np.zeros((*reference_rel.shape, time_grid.size))
for file in snakemake.input.true_site_relatedness:
    true_rel += np.load(file) / num_replicates

infr_rel = np.zeros((*reference_rel.shape, time_grid.size))
for file in snakemake.input.infr_site_relatedness:
    infr_rel += np.load(file) / num_replicates

# plot relatedness per individual pair
stride = int(np.ceil(num_individuals / max_individuals))
subset = np.arange(0, num_individuals, stride)
cols = subset.size
rows = subset.size
fig, axs = plt.subplots(
    rows, cols, 
    figsize=(1.25 * cols, 1 * rows), 
    sharex=True, sharey=True, 
    squeeze=False,
)
scale = 1e-3
midpoints = (time_grid[:-1] + time_grid[1:]) / 2
for i, ii in enumerate(subset):
    for j, jj in enumerate(subset):
        if j < i:
            axs[i, j].set_visible(False)
        else:
            label = f"({ii}, {jj})"
            true = true_rel[ii, jj].cumsum()
            infr = infr_rel[ii, jj].cumsum()
            axs[i, j].text(0.05, 0.95, label, ha="left", va="top", transform=axs[i, j].transAxes, size=8)
            axs[i, j].plot(time_grid, scale * true, "-", color="black", label="true", linewidth=1)
            axs[i, j].plot(time_grid, scale * infr, "-", color="firebrick", label="estimated", linewidth=1)
            axs[i, j].set_xscale("log")
            if snakemake.params.log_relatedness:
                axs[i, j].set_yscale("log")
            if i == j:
                axs[i, j].tick_params(axis="both", labelbottom=True, labelleft=True)
                #axs[i, j].ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
fig.supylabel("Number of shared derived mutations (thousands)")
fig.supxlabel("Maximum mutation age")
fig.legend(
    *axs[0, 0].get_legend_handles_labels(), 
    loc="lower left",
    bbox_to_anchor=(0.25, 0.25),
)
fig.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.savefig(snakemake.output.rel_plot)
plt.clf()
