"""
Plot the proportion of sites that are correctly/incorrectly repolarised
(ancestral state switched relative to input), as a function of true ancestral
frequency (AFS bin).

Part of https://github.com/nspope/singer-snakemake.
"""

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


matplotlib.rcParams["figure.dpi"] = 300


mispolarised = sum([np.load(f) for f in snakemake.input.mispolarised])
afs_dims = mispolarised.shape[2:]
afs_bins = list(itertools.product(*[range(d) for d in afs_dims]))
afs_labels = [str(x).replace(" ", "") for x in afs_bins]

bar_width = 1.0
fig_width = max(5, len(afs_bins) * 0.03)
fig, axs = plt.subplots(
    3, figsize=(fig_width, 6), 
    constrained_layout=True, 
    sharex=True, sharey=True,
)

if len(afs_dims) > 1:
    tick_locations, tick_labels = zip(
        *((i, x) for i, x in enumerate(afs_labels) if x.endswith(",0)"))
    )
else:
    tick_locations = np.arange(len(afs_bins), step=5)
    tick_labels = tick_locations

axs[0].set_title("All sites")
denom = mispolarised.sum(axis=(0, 1))
denom = np.array([denom[*f] for f in afs_bins])
bottom = np.zeros(len(afs_labels))
for j, tree_label in enumerate(["correct", "incorrect"]):
    for i, vcf_label in enumerate(["correct", "incorrect"]):
        label = f"{vcf_label} in vcf, {tree_label} in trees"
        count = np.array([mispolarised[i, j][*f] for f in afs_bins]) / denom
        color = "firebrick" if tree_label == "incorrect" else "dodgerblue"
        alpha = 1.0 if vcf_label == tree_label else 0.5
        axs[0].bar(afs_labels, count, bottom=bottom, label=label, width=bar_width, color=color, alpha=alpha)
        bottom += count

axs[1].set_title("Sites with incorrect ancestral state in input")
denom = mispolarised[1, ...].sum(axis=0)
denom = np.array([denom[*f] for f in afs_bins])
bottom = np.zeros(len(afs_labels))
i, vcf_label = 1, "incorrect"
for j, tree_label in enumerate(["correct", "incorrect"]):
    label = f"{vcf_label} in vcf, {tree_label} in trees"
    count = np.array([mispolarised[i, j][*f] for f in afs_bins]) / denom
    color = "firebrick" if tree_label == "incorrect" else "dodgerblue"
    alpha = 1.0 if vcf_label == tree_label else 0.5
    axs[1].bar(afs_labels, count, bottom=bottom, label=label, width=bar_width, color=color, alpha=alpha)
    bottom += count

axs[2].set_title("Sites with correct ancestral state in input")
denom = mispolarised[0, ...].sum(axis=0)
denom = np.array([denom[*f] for f in afs_bins])
bottom = np.zeros(len(afs_labels))
i, vcf_label = 0, "correct"
for j, tree_label in enumerate(["correct", "incorrect"]):
    label = f"{vcf_label} in vcf, {tree_label} in trees"
    count = np.array([mispolarised[i, j][*f] for f in afs_bins]) / denom
    color = "firebrick" if tree_label == "incorrect" else "dodgerblue"
    alpha = 1.0 if vcf_label == tree_label else 0.5
    axs[2].bar(afs_labels, count, bottom=bottom, label=label, width=bar_width, color=color, alpha=alpha)
    bottom += count

axs[2].set_xlim(-0.5, len(afs_bins) - 0.5)
axs[2].set_ylim(0, 1)
axs[2].set_xticks(tick_locations)
axs[2].set_xticklabels(tick_labels, rotation=90, ha="center", size=8)
axs[2].xaxis.set_minor_locator(FixedLocator(np.arange(len(afs_bins))))
fig.legend(
    *axs[0].get_legend_handles_labels(), 
    loc='outside upper center', 
    ncol=2, fontsize=8, frameon=False,
)
fig.supxlabel("True frequency of ancestral state (AFS bin)")
fig.supylabel("Proportion")

plt.savefig(snakemake.output.pol_plot)
plt.clf()
