"""
Plot distribution of mutation ages, using the down-projected time-windowed
allele frequency spectrum.
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FixedLocator


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
for f, ax in zip(bins, axs.ravel()):
    true_pdf = true_afs[*f] / true_afs[*f].sum()
    infr_pdf = infr_afs[*f] / infr_afs[*f].sum()
    ax.step(time_grid[1:], true_pdf[1:], where="post", color="black", label="true")
    ax.step(time_grid[1:], infr_pdf[1:], where="post", color="firebrick", label="estimated")
    ax.text(0.05, 0.95, f"{f}", ha="left", va="top", size=8, transform=ax.transAxes)
    ax.set_xscale("log")
fig.supylabel("Proportion of mutations")
fig.supxlabel("Mutation age")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='outside upper center', ncol=4)
plt.savefig(snakemake.output.pdf_plot)
plt.clf()

# plot mean per AFS bin in single panel
fig, axs = plt.subplots(1, figsize=(len(bins) * 0.05, 4), constrained_layout=True)
tick_labels = [str(x).replace(" ", "") for x in bins]
tick_locations, tick_labels = zip(
    *((i, x) for i, x in enumerate(tick_labels) if x.endswith(",0)"))
)
midpoints = (time_grid[:-1] + time_grid[1:]) / 2
true_mn = []
infr_mn = []
for f in bins:
    true_pdf = true_afs[*f] / true_afs[*f].sum()
    infr_pdf = infr_afs[*f] / infr_afs[*f].sum()
    true_mn.append(np.sum(true_pdf[:-1] / true_pdf[:-1].sum() * midpoints))
    infr_mn.append(np.sum(infr_pdf[:-1] / infr_pdf[:-1].sum() * midpoints))
axs.plot(np.arange(len(bins)), true_mn, "o", color="black", markersize=4, label="true")
axs.plot(np.arange(len(bins)), infr_mn, "o", color="firebrick", markersize=4, label="estimated")
axs.set_xticks(tick_locations)
axs.set_xticklabels(tick_labels, rotation=90, ha="center", size=8)
axs.xaxis.set_minor_locator(FixedLocator(np.arange(len(bins))))
axs.set_yscale("log")
axs.set_ylabel("E[mutation age]")
axs.set_xlabel("AFS bin")
axs.legend()
plt.savefig(snakemake.output.exp_plot)
plt.clf()





