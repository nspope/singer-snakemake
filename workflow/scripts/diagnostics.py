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
import matplotlib
from datetime import datetime

warnings.simplefilter("ignore")
matplotlib.rcParams["figure.dpi"] = 300

# --- lib --- #

def tag(): 
    return f"[singer-snakemake::{snakemake.rule}::{str(datetime.now())}]"


# --- implm --- #

mcmc_thin = snakemake.params.mcmc_thin
polarised = snakemake.params.polarised
windows = pickle.load(open(snakemake.input.windows, "rb")).position
coord = windows[:-1] / 2 + windows[1:] / 2 

# statistics from VCF
vcf_stats = pickle.load(open(snakemake.input.vcf_stats, "rb"))
site_diversity = vcf_stats['diversity']
site_tajima_d = vcf_stats['tajima_d']
site_afs = vcf_stats['afs']

# statistics from MCMC samples of trees
num_mcmc = snakemake.params.mcmc_samples
num_burnin = snakemake.params.mcmc_burnin
mcmc_iterates = np.arange(num_mcmc) * mcmc_thin
repolarised = []
expected_load = []
observed_load = []
branch_diversity = []
branch_tajima_d = []
branch_afs = []
for i, stats_file in enumerate(snakemake.input.stats):
    stats = pickle.load(open(stats_file, "rb"))
    repolarised.append(stats['repolarised'])
    expected_load.append(stats['expected_load'])
    observed_load.append(stats['observed_load'])
    branch_diversity.append(stats['diversity'])
    branch_tajima_d.append(stats['tajima_d'])
    branch_afs.append(stats['afs'])
repolarised = np.array(repolarised)
expected_load = np.stack(expected_load, axis=-1)
observed_load = np.stack(observed_load, axis=-1)
branch_diversity = np.stack(branch_diversity, axis=-1)
branch_tajima_d = np.stack(branch_tajima_d, axis=-1)
branch_afs = np.stack(branch_afs, axis=-1)[:site_afs.size]

# posterior mean, quantiles
conf_level = snakemake.params.confidence_level
quantiles = np.array([conf_level / 2, 1 - conf_level / 2])
trace_diversity = np.nanmean(branch_diversity, axis=0)  # NB: not correctly weighting combination, fine for trace
trace_tajima_d = np.nanmean(branch_tajima_d, axis=0)
quant_expected_load = np.quantile(expected_load[:, num_burnin:], quantiles, axis=-1)
quant_observed_load = np.quantile(observed_load[:, num_burnin:], quantiles, axis=-1)
quant_diversity = np.nanquantile(branch_diversity[:, num_burnin:], quantiles, axis=-1)
quant_tajima_d = np.nanquantile(branch_tajima_d[:, num_burnin:], quantiles, axis=-1)
quant_afs = np.quantile(branch_afs[:, num_burnin:], quantiles, axis=-1)
mean_expected_load = np.mean(expected_load[:, num_burnin:], axis=-1)
mean_observed_load = np.mean(observed_load[:, num_burnin:], axis=-1)
mean_diversity = np.nanmean(branch_diversity[:, num_burnin:], axis=-1)
mean_tajima_d = np.nanmean(branch_tajima_d[:, num_burnin:], axis=-1)
mean_afs = np.mean(branch_afs[:, num_burnin:], axis=-1)

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
    plt.plot(mcmc_iterates, trace_diversity, "-", c='firebrick')
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[diversity]")
    plt.tight_layout()
    plt.savefig(snakemake.output.diversity_trace)
    plt.clf()

if snakemake.output.tajima_d_trace is not None:
    plt.figure(figsize=(5, 4))
    plt.plot(mcmc_iterates, trace_tajima_d, "-", c='firebrick')
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[Tajima's D]")
    plt.tight_layout()
    plt.savefig(snakemake.output.tajima_d_trace)
    plt.clf()

if snakemake.output.repolarised_trace is not None:
    plt.figure(figsize=(5, 4))
    plt.plot(mcmc_iterates, repolarised, "-", c="firebrick")
    plt.xlabel("MCMC iteration")
    plt.ylabel("Proportion flipped ancestral states")
    plt.tight_layout()
    plt.savefig(snakemake.output.repolarised_trace)
    plt.clf()

if snakemake.output.mutational_load_trace is not None:
    plt.figure(figsize=(5, 4))
    for i, x in enumerate(observed_load):
        plt.plot(mcmc_iterates, x, "-", color="black", linewidth=0.5, alpha=0.25)
    plt.xlabel("MCMC iteration")
    plt.ylabel("Derived mutations / base in each sample")
    plt.tight_layout()
    plt.savefig(snakemake.output.mutational_load_trace)
    plt.clf()

if snakemake.output.diversity_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_diversity[0], quant_diversity[1], color="firebrick", alpha=0.1)
    plt.plot(coord, mean_diversity, "-o", c="firebrick", label="branch-ARG", markersize=3)
    plt.plot(coord, site_diversity, "-o", c="black", label="site-VCF", markersize=3)
    plt.xlabel("Position on chromosome")
    plt.ylabel("Diversity / base")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.diversity_skyline)
    plt.clf()

if snakemake.output.tajima_d_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_tajima_d[0], quant_tajima_d[1], color='firebrick', alpha=0.1)
    plt.plot(coord, mean_tajima_d, "-o", c='firebrick', label='branch-ARG', markersize=3)
    plt.plot(coord, site_tajima_d, "-o", c='black', label='site-VCF', markersize=3)
    plt.xlabel("Position on chromosome")
    plt.ylabel("Tajima's D")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.tajima_d_skyline)
    plt.clf()

if snakemake.output.mutational_load is not None:
    samples = np.arange(mean_expected_load.size)
    plt.figure(figsize=(max(5, 0.1 * samples.size), 4))
    plt.axhline(y=mean_expected_load.mean(), color='firebrick', linestyle="dashed", label="expected")
    plt.plot(samples, mean_observed_load, "o", color='black', markersize=2)
    plt.vlines(samples, *quant_observed_load, color="black", label="observed")
    plt.xlabel("Sample ID")
    plt.ylabel("Number of derived mutations / base")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.mutational_load)
    plt.clf()

if snakemake.output.frequency_spectrum is not None:
    # TODO: add fold change relative to observed spectrum
    plt.figure(figsize=(8, 4))
    freq = np.arange(1, site_afs.size)
    plt.fill_between(freq, quant_afs[0][1:], quant_afs[1][1:], color='firebrick', alpha=0.1)
    plt.scatter(freq, mean_afs[1:], c='firebrick', label='branch-ARG', s=8)
    plt.scatter(freq, site_afs[1:], c='black', label='site-VCF', s=8)
    plt.xlabel("Derived allele frequency" if polarised else "Minor allele frequency")
    plt.ylabel("# of variants / base")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.frequency_spectrum)
    plt.clf()


# TODO: clean this up
# stratified summary stats
if snakemake.params.stratify is not None:
    vcf_strata_stats = pickle.load(open(snakemake.input.vcf_strata_stats, "rb"))
    site_strata_divergence = vcf_strata_stats['divergence']
    site_strata_afs = vcf_strata_stats['afs']
    strata = vcf_strata_stats["strata"]

    # statistics from MCMC samples of trees
    branch_strata_divergence = np.zeros((*site_strata_divergence.shape, num_mcmc))
    branch_strata_afs = [np.zeros((*x.shape, num_mcmc)) for x in site_strata_afs]
    for i, stats_file in enumerate(snakemake.input.strata_stats):
        strata_stats = pickle.load(open(stats_file, "rb"))
        assert strata_stats["strata"] == strata
        branch_strata_divergence[..., i] = strata_stats['divergence']
        for j, x in enumerate(strata_stats['afs']):
            branch_strata_afs[j][:, i] = x[:site_strata_afs[j].size]

    # posterior mean, quantiles
    trace_strata_divergence = np.nanmean(branch_strata_divergence, axis=0)
    quant_strata_divergence = np.nanquantile(
        branch_strata_divergence[..., num_burnin:], 
        quantiles, 
        axis=-1,
    )
    quant_strata_afs = [
        np.quantile(x[..., num_burnin:], quantiles, axis=1) 
        for x in branch_strata_afs
    ]
    mean_strata_divergence = np.nanmean(
        branch_strata_divergence[..., num_burnin:], 
        axis=-1,
    )
    mean_strata_afs = [
        np.mean(x[..., num_burnin:], axis=-1) 
        for x in branch_strata_afs
    ]

    # plot chunk divergence
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
                axs[i, j].set_title(f"{p} vs {q}", loc="left", size=12)
                k += 1
            else:
                axs[i, j].set_visible(False)
    fig.supxlabel("Site (VCF) divergence per chunk")
    fig.supylabel("E[divergence] per chunk")
    plt.savefig(strata_divergence_scatter)
    plt.clf()

    # plot afs across strata
    strata_afs = os.path.join(
        os.path.dirname(snakemake.output.frequency_spectrum),
        "strata-frequency-spectrum.png",
    )
    fig, axs = plt.subplots(
        len(strata), 1, figsize=(8, len(strata) * 4),
        constrained_layout=True, squeeze=False,
    )
    k = 0
    for i, p in enumerate(strata):
        freq = np.arange(1, site_strata_afs[i].size)
        axs[i, 0].fill_between(
            freq, 
            quant_strata_afs[i][0, 1:], 
            quant_strata_afs[i][1, 1:], 
            color='firebrick', 
            alpha=0.1,
        )
        axs[i, 0].scatter(
            freq, mean_strata_afs[i][1:], 
            c='firebrick', label='branch-ARG', s=8,
        )
        axs[i, 0].scatter(
            freq, site_strata_afs[i][1:], 
            c='black', label='site-VCF', s=8,
        )
        axs[i, 0].set_title(f"{p}", loc="left", size=12)
        axs[i, 0].set_yscale("log")
        axs[i, 0].legend()
    fig.supxlabel(
        "Derived allele frequency" if polarised 
        else "Minor allele frequency"
    )
    fig.supylabel("# of variants / base")
    plt.savefig(strata_afs)
    plt.clf()
    
