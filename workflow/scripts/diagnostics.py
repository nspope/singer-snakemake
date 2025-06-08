"""
Plot "fitted" vs observed statistics from the ARG.

Part of https://github.com/nspope/singer-snakemake.
"""

import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import msprime
import warnings
from datetime import datetime


# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

warnings.simplefilter('ignore')

# statistics from VCF
ratemap = pickle.load(open(snakemake.input.ratemap, "rb"))
windows = ratemap.position
coord = windows[:-1] / 2 + windows[1:] / 2 
vcf_stats = pickle.load(open(snakemake.input.vcf_stats, "rb"))
site_diversity = vcf_stats['diversity']
site_tajima_d = vcf_stats['tajima_d']
site_folded_afs = vcf_stats['folded_afs']
site_unfolded_afs = vcf_stats['unfolded_afs']

# statistics from MCMC samples of trees
num_mcmc = snakemake.params.mcmc_samples
num_burnin = snakemake.params.mcmc_burnin
branch_diversity = np.zeros((windows.size - 1, num_mcmc))
branch_tajima_d = np.zeros((windows.size - 1, num_mcmc))
branch_folded_afs = np.zeros((site_folded_afs.size, num_mcmc))
branch_unfolded_afs = np.zeros((site_unfolded_afs.size, num_mcmc))
for i, stats_file in enumerate(snakemake.input.stats):
    stats = pickle.load(open(stats_file, "rb"))
    branch_diversity[:, i] = stats['diversity']
    branch_tajima_d[:, i] = stats['tajima_d']
    branch_folded_afs[:, i] = stats['folded_afs'][:site_folded_afs.size]
    branch_unfolded_afs[:, i] = stats['unfolded_afs'][:site_unfolded_afs.size]

# posterior mean, quantiles
conf_level = snakemake.params.confidence_level
quantiles = np.array([conf_level / 2, 1 - conf_level / 2])
trace_diversity = np.nanmean(branch_diversity, axis=0)  # NB: not correctly weighting combination, fine for trace
trace_tajima_d = np.nanmean(branch_tajima_d, axis=0)
quant_diversity = np.nanquantile(branch_diversity[:, num_burnin:], quantiles, axis=-1)
quant_tajima_d = np.nanquantile(branch_tajima_d[:, num_burnin:], quantiles, axis=-1)
quant_folded_afs = np.quantile(branch_folded_afs[:, num_burnin:], quantiles, axis=-1)
quant_unfolded_afs = np.quantile(branch_unfolded_afs[:, num_burnin:], quantiles, axis=-1)
mean_diversity = np.nanmean(branch_diversity[:, num_burnin:], axis=1)
mean_tajima_d = np.nanmean(branch_tajima_d[:, num_burnin:], axis=1)
mean_folded_afs = np.mean(branch_folded_afs[:, num_burnin:], axis=1)
mean_unfolded_afs = np.mean(branch_unfolded_afs[:, num_burnin:], axis=1)

if snakemake.output.diversity_scatter is not None:
    plt.figure(figsize=(5, 4))
    plt.scatter(site_diversity, mean_diversity, c='firebrick', s=8)
    plt.axline((np.nanmean(site_diversity), np.nanmean(site_diversity)), slope=1, color='black', linestyle="dashed")
    plt.xlabel("Site (VCF) diversity per chunk")
    plt.ylabel("E[diversity] per chunk")
    plt.tight_layout()
    plt.savefig(snakemake.output.diversity_scatter)
    plt.clf()

if snakemake.output.tajima_d_scatter is not None:
    plt.figure(figsize=(5, 4))
    plt.scatter(site_tajima_d, mean_tajima_d, c='firebrick', s=8)
    plt.axline((np.nanmean(site_tajima_d), np.nanmean(site_tajima_d)), slope=1, color='black', linestyle="dashed")
    plt.xlabel("Site (VCF) Tajima's D per chunk")
    plt.ylabel("E[Tajima's D] per chunk")
    plt.tight_layout()
    plt.savefig(snakemake.output.tajima_d_scatter)
    plt.clf()

if snakemake.output.diversity_trace is not None:
    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(trace_diversity.size), trace_diversity, "-", c='firebrick')
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[diversity]")
    plt.tight_layout()
    plt.savefig(snakemake.output.diversity_trace)
    plt.clf()

if snakemake.output.tajima_d_trace is not None:
    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(trace_tajima_d.size), trace_tajima_d, "-", c='firebrick')
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[Tajima's D]")
    plt.tight_layout()
    plt.savefig(snakemake.output.tajima_d_trace)
    plt.clf()

if snakemake.output.diversity_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_diversity[0], quant_diversity[1], color='firebrick', alpha=0.1)
    plt.plot(coord, mean_diversity, "-", c='firebrick', label='branch-ARG')
    plt.plot(coord, site_diversity, "-", c='black', label='site-VCF')
    plt.xlabel("Position on chromosome")
    plt.ylabel("Diversity / base")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.diversity_skyline)
    plt.clf()

if snakemake.output.tajima_d_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_tajima_d[0], quant_tajima_d[1], color='firebrick', alpha=0.1)
    plt.plot(coord, mean_tajima_d, "-", c='firebrick', label='branch-ARG')
    plt.plot(coord, site_tajima_d, "-", c='black', label='site-VCF')
    plt.xlabel("Position on chromosome")
    plt.ylabel("Tajima's D")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.tajima_d_skyline)
    plt.clf()

if snakemake.output.folded_afs is not None:
    plt.figure(figsize=(8, 4))
    freq = np.arange(1, site_folded_afs.size)
    plt.fill_between(freq, quant_folded_afs[0][1:], quant_folded_afs[1][1:], color='firebrick', alpha=0.1)
    plt.scatter(freq, mean_folded_afs[1:], c='firebrick', label='branch-ARG', s=8)
    plt.scatter(freq, site_folded_afs[1:], c='black', label='site-VCF', s=8)
    plt.xlabel("Minor allele frequency")
    plt.ylabel("# of variants / base")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.folded_afs)
    plt.clf()

if snakemake.output.unfolded_afs is not None:
    plt.figure(figsize=(8, 4))
    freq = np.arange(1, site_unfolded_afs.size - 1)
    plt.fill_between(freq, quant_unfolded_afs[0][1:-1], quant_unfolded_afs[1][1:-1], color='firebrick', alpha=0.1)
    plt.scatter(freq, mean_unfolded_afs[1:-1], c='firebrick', label='branch-ARG', s=8)
    plt.scatter(freq, site_unfolded_afs[1:-1], c='black', label='site-VCF', s=8)
    plt.xlabel("Derived allele frequency")
    plt.ylabel("# of variants / base")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.unfolded_afs)
    plt.clf()


# TODO: clean this up
# stratified summary stats
if snakemake.params.stratify is not None:

    # statistics from VCF
    vcf_strata_stats = pickle.load(open(snakemake.input.vcf_strata_stats, "rb"))
    site_strata_divergence = vcf_strata_stats['divergence']
    site_strata_folded_afs = vcf_strata_stats['folded_afs']
    site_strata_unfolded_afs = vcf_strata_stats['unfolded_afs']
    strata = vcf_strata_stats["strata"]

    # statistics from MCMC samples of trees
    branch_strata_divergence = np.zeros((*site_strata_divergence.shape, num_mcmc))
    branch_strata_folded_afs = [np.zeros((*x.shape, num_mcmc)) for x in site_strata_folded_afs]
    branch_strata_unfolded_afs = [np.zeros((*x.shape, num_mcmc)) for x in site_strata_unfolded_afs]
    for i, stats_file in enumerate(snakemake.input.strata_stats):
        strata_stats = pickle.load(open(stats_file, "rb"))
        assert strata_stats["strata"] == strata
        branch_strata_divergence[..., i] = strata_stats['divergence']
        for j, x in enumerate(strata_stats['folded_afs']):
            branch_strata_folded_afs[j][:, i] = x[:site_strata_folded_afs[j].size]
        for j, x in enumerate(strata_stats['unfolded_afs']):
            branch_strata_unfolded_afs[j][:, i] = x[:site_strata_unfolded_afs[j].size]

    # posterior mean, quantiles
    trace_strata_divergence = np.nanmean(branch_strata_divergence, axis=0)
    quant_strata_divergence = np.nanquantile(branch_strata_divergence[..., num_burnin:], quantiles, axis=-1)
    quant_strata_folded_afs = [
        np.quantile(x[..., num_burnin:], quantiles, axis=1) for x in branch_strata_folded_afs
    ]
    quant_strata_unfolded_afs = [
        np.quantile(x[..., num_burnin:], quantiles, axis=1) for x in branch_strata_unfolded_afs
    ]
    mean_strata_divergence = np.nanmean(branch_strata_divergence[..., num_burnin:], axis=-1)
    mean_strata_folded_afs = [
        np.mean(x[..., num_burnin:], axis=-1) for x in branch_strata_folded_afs
    ]
    mean_strata_unfolded_afs = [
        np.mean(x[..., num_burnin:], axis=-1) for x in branch_strata_unfolded_afs
    ]

    # plot divergence
    strata_divergence_scatter = os.path.join(
        os.path.dirname(snakemake.output.diversity_scatter), 
        "strata-divergence-scatter.png",
    )
    fig, axs = plt.subplots(
        len(strata), len(strata), 
        figsize=(len(strata) * 4, len(strata) * 3.5),
        constrained_layout=True, squeeze=False,
    )
    k = 0
    for i, p in enumerate(strata):
        for j, q in enumerate(strata):
            if j >= i:
                axs[i, j].scatter(
                    site_strata_divergence[:, k], 
                    mean_strata_divergence[:, k], 
                    c='firebrick', s=8,
                )
                x = np.nanmean(site_strata_divergence[:, k])
                axs[i, j].axline((x, x), slope=1, color='black', linestyle="dashed")
                axs[i, j].set_title(f"{p} - {q}", loc="left", size=12)
                k += 1
            else:
                axs[i, j].set_visible(False)
    fig.supxlabel("Site (VCF) divergence per chunk")
    fig.supylabel("E[divergence] per chunk")
    plt.savefig(strata_divergence_scatter)
    plt.clf()

    # plot folded afs across strata
    strata_folded_afs = os.path.join(
        os.path.dirname(snakemake.output.folded_afs),
        "strata-folded-afs.png",
    )
    fig, axs = plt.subplots(
        len(strata), 1, figsize=(8, len(strata) * 4),
        constrained_layout=True, squeeze=False,
    )
    k = 0
    for i, p in enumerate(strata):
        freq = np.arange(1, site_strata_folded_afs[i].size)
        axs[i, 0].fill_between(
            freq, 
            quant_strata_folded_afs[i][0, 1:], 
            quant_strata_folded_afs[i][1, 1:], 
            color='firebrick', 
            alpha=0.1,
        )
        axs[i, 0].scatter(
            freq, mean_strata_folded_afs[i][1:], 
            c='firebrick', label='branch-ARG', s=8,
        )
        axs[i, 0].scatter(
            freq, site_strata_folded_afs[i][1:], 
            c='black', label='site-VCF', s=8,
        )
        axs[i, 0].set_title(f"{p}", loc="left", size=12)
        axs[i, 0].set_yscale("log")
        axs[i, 0].legend()
    fig.supxlabel("Minor allele frequency")
    fig.supylabel("# of variants / base")
    plt.savefig(strata_folded_afs)
    plt.clf()
    
    # plot unfolded afs across strata
    strata_unfolded_afs = os.path.join(
        os.path.dirname(snakemake.output.unfolded_afs),
        "strata-unfolded-afs.png",
    )
    fig, axs = plt.subplots(
        len(strata), 1, figsize=(8, len(strata) * 4),
        constrained_layout=True, squeeze=False,
    )
    k = 0
    for i, p in enumerate(strata):
        freq = np.arange(1, site_strata_unfolded_afs[i].size - 1)
        axs[i, 0].fill_between(
            freq, 
            quant_strata_unfolded_afs[i][0, 1:-1], 
            quant_strata_unfolded_afs[i][1, 1:-1], 
            color='firebrick', 
            alpha=0.1,
        )
        axs[i, 0].scatter(
            freq, mean_strata_unfolded_afs[i][1:-1], 
            c='firebrick', label='branch-ARG', s=8,
        )
        axs[i, 0].scatter(
            freq, site_strata_unfolded_afs[i][1:-1], 
            c='black', label='site-VCF', s=8,
        )
        axs[i, 0].set_title(f"{p}", loc="left", size=12)
        axs[i, 0].set_yscale("log")
        axs[i, 0].legend()
    fig.supxlabel("Derived allele frequency")
    fig.supylabel("# of variants / base")
    plt.savefig(strata_unfolded_afs)
    plt.clf()
    
