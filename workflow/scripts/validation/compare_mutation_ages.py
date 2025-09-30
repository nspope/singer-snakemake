"""
Plot distribution of mutation ages, using the down-projected time-windowed
allele frequency spectrum.
"""

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


matplotlib.rcParams["figure.dpi"] = 300

time_grid = np.append(0, snakemake.params.time_grid[:-1])
reference_afs = np.load(next(iter(snakemake.input.true_site_afs)))[..., 0]
bins = list(itertools.product(*[np.arange(dim).tolist() for dim in reference_afs.shape]))
num_replicates = len(snakemake.input.true_site_afs)

# average over simulations
true_afs = np.zeros((*reference_afs.shape, time_grid.size))
for file in snakemake.input.true_site_afs:
    true_afs += np.load(file) / num_replicates

infr_afs = np.zeros((*reference_afs.shape, time_grid.size))
for file in snakemake.input.infr_site_afs:
    infr_afs += np.load(file) / num_replicates

# remove last bin (ending at time infinity)
true_marginal = true_afs.sum(axis=-1)
infr_marginal = infr_afs.sum(axis=-1)
true_afs = true_afs[..., :-1]
infr_afs = infr_afs[..., :-1]

# plot mutation age distribution per AFS bin
cols = 9
rows = int(np.ceil(len(bins) / cols))
fig, axs = plt.subplots(
    rows, cols, 
    figsize=(1.25 * cols, 1 * rows), 
    sharex=True, sharey=True, 
    constrained_layout=True,
    squeeze=False,
)
midpoints = (time_grid[:-1] + time_grid[1:]) / 2
for f, ax in zip(bins, axs.ravel()):
    # FIXME: not plotting the first bin here given log time measure
    true_pdf = true_afs[*f] / true_afs[*f].sum()
    infr_pdf = infr_afs[*f] / infr_afs[*f].sum()
    ax.plot(midpoints[1:], true_pdf[1:], "-", color="black", label="true", linewidth=1)
    ax.plot(midpoints[1:], infr_pdf[1:], "-", color="firebrick", label="inferred", linewidth=1)
    ax.set_xscale("log")
    ax.text(0.05, 0.95, f"{f}", ha="left", va="top", size=8, transform=ax.transAxes)
fig.supylabel("Proportion of mutations")
fig.supxlabel("Mutation age")
fig.legend(
    *axs[0, 0].get_legend_handles_labels(), 
    loc='outside upper center', 
    ncol=2, frameon=False,
)
plt.savefig(snakemake.output.pdf_plot)
plt.clf()

# plot proportion of sites and mean age per AFS bin in single panels
width = max(3.5, len(bins) * 0.03)
fig, axs = plt.subplots(2, 1, figsize=(width, 6), constrained_layout=True, sharex=True)

if reference_afs.ndim > 1:
    tick_labels = [str(x).replace(" ", "") for x in bins]
    tick_locations, tick_labels = zip(
        *((i, x) for i, x in enumerate(tick_labels) if x.endswith(",0)"))
    )
else:
    tick_locations = np.arange(len(bins), step=5)
    tick_labels = tick_locations

true_mean = []
infr_mean = []
true_prop = []
infr_prop = []
for f in bins:
    true_pdf = true_afs[*f] / true_afs[*f].sum()
    infr_pdf = infr_afs[*f] / infr_afs[*f].sum()
    true_mean.append(np.sum(true_pdf / true_pdf.sum() * midpoints))
    infr_mean.append(np.sum(infr_pdf / infr_pdf.sum() * midpoints))
    true_prop.append(true_marginal[*f] / true_marginal.sum())
    infr_prop.append(infr_marginal[*f] / infr_marginal.sum())
axs[0].plot(np.arange(len(bins)), true_prop, "o", color="black", markersize=2, label="true")
axs[0].plot(np.arange(len(bins)), infr_prop, "o", color="firebrick", markersize=2, label="inferred")
axs[0].set_ylabel("Proportion of mutations")
axs[0].set_yscale("log")
axs[1].plot(np.arange(len(bins)), true_mean, "o", color="black", markersize=2, label="true")
axs[1].plot(np.arange(len(bins)), infr_mean, "o", color="firebrick", markersize=2, label="inferred")
axs[1].set_yscale("log")
axs[1].set_ylabel("Mean mutation age")

axs[1].set_xlim(-1, len(bins))
axs[1].set_xticks(tick_locations)
axs[1].set_xticklabels(tick_labels, rotation=90, ha="center", size=8)
axs[1].xaxis.set_minor_locator(FixedLocator(np.arange(len(bins))))
axs[1].set_xlabel("Frequency of derived state (AFS bin)")
fig.legend(
    *axs[0].get_legend_handles_labels(), 
    loc='outside upper center', 
    ncol=2, frameon=False,
)

plt.savefig(snakemake.output.exp_plot)
plt.clf()

