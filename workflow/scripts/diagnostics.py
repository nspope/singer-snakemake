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
windows = pickle.load(open(snakemake.input.windows, "rb")).position
coord = windows[:-1] / 2 + windows[1:] / 2 

# statistics from MCMC samples of trees
num_mcmc = snakemake.params.mcmc_samples
num_burnin = snakemake.params.mcmc_burnin
mcmc_iterates = np.arange(num_mcmc) * mcmc_thin

expected_load = []
expected_diversity = []
expected_segsites = []
expected_tajima_d = []
expected_afs = []
for i, expected_stats_file in enumerate(snakemake.input.expected_stats):
    expected_stats = pickle.load(open(expected_stats_file, "rb"))
    expected_load.append(expected_stats['load'])
    expected_diversity.append(expected_stats['diversity'])
    expected_segsites.append(expected_stats['segsites'])
    expected_tajima_d.append(expected_stats['tajima_d'])
    expected_afs.append(expected_stats['afs'])
expected_load = np.stack(expected_load, axis=-1)
expected_diversity = np.stack(expected_diversity, axis=-1)
expected_segsites = np.stack(expected_segsites, axis=-1)
expected_tajima_d = np.stack(expected_tajima_d, axis=-1)
expected_afs = np.stack(expected_afs, axis=-1)

repolarised = []
multimapped = []
observed_load = []
observed_diversity = []
observed_segsites = []
observed_tajima_d = []
observed_afs = []
for i, observed_stats_file in enumerate(snakemake.input.observed_stats):
    observed_stats = pickle.load(open(observed_stats_file, "rb"))
    observed_load.append(observed_stats['load'])
    observed_diversity.append(observed_stats['diversity'])
    observed_segsites.append(observed_stats['segsites'])
    observed_tajima_d.append(observed_stats['tajima_d'])
    observed_afs.append(observed_stats['afs'])
    repolarised.append(observed_stats['repolarised'])
    multimapped.append(observed_stats['multimapped'])
multimapped = np.array(multimapped)
repolarised = np.array(repolarised)
observed_load = np.stack(observed_load, axis=-1)
observed_diversity = np.stack(observed_diversity, axis=-1)
observed_segsites = np.stack(observed_segsites, axis=-1)
observed_tajima_d = np.stack(observed_tajima_d, axis=-1)
observed_afs = np.stack(observed_afs, axis=-1)

# posterior mean, quantiles
conf_level = snakemake.params.confidence_level
quantiles = np.array([conf_level / 2, 1 - conf_level / 2])

trace_expected_diversity = np.nanmean(expected_diversity, axis=0)
trace_expected_segsites = np.nanmean(expected_segsites, axis=0)  
trace_expected_tajima_d = np.nanmean(expected_tajima_d, axis=0)
quant_expected_load = np.quantile(expected_load[:, num_burnin:], quantiles, axis=-1)
quant_expected_diversity = np.nanquantile(expected_diversity[:, num_burnin:], quantiles, axis=-1)
quant_expected_segsites = np.nanquantile(expected_segsites[:, num_burnin:], quantiles, axis=-1)
quant_expected_tajima_d = np.nanquantile(expected_tajima_d[:, num_burnin:], quantiles, axis=-1)
quant_expected_afs = np.quantile(expected_afs[:, num_burnin:], quantiles, axis=-1)
mean_expected_load = np.mean(expected_load[:, num_burnin:], axis=-1)
mean_expected_diversity = np.nanmean(expected_diversity[:, num_burnin:], axis=-1)
mean_expected_segsites = np.nanmean(expected_segsites[:, num_burnin:], axis=-1)
mean_expected_tajima_d = np.nanmean(expected_tajima_d[:, num_burnin:], axis=-1)
mean_expected_afs = np.mean(expected_afs[:, num_burnin:], axis=-1)

trace_observed_diversity = np.nanmean(observed_diversity, axis=0)
trace_observed_segsites = np.nanmean(observed_segsites, axis=0) 
trace_observed_tajima_d = np.nanmean(observed_tajima_d, axis=0)
quant_observed_load = np.quantile(observed_load[:, num_burnin:], quantiles, axis=-1)
quant_observed_diversity = np.nanquantile(observed_diversity[:, num_burnin:], quantiles, axis=-1)
quant_observed_segsites = np.nanquantile(observed_segsites[:, num_burnin:], quantiles, axis=-1)
quant_observed_tajima_d = np.nanquantile(observed_tajima_d[:, num_burnin:], quantiles, axis=-1)
quant_observed_afs = np.quantile(observed_afs[:, num_burnin:], quantiles, axis=-1)
mean_observed_load = np.mean(observed_load[:, num_burnin:], axis=-1)
mean_observed_diversity = np.nanmean(observed_diversity[:, num_burnin:], axis=-1)
mean_observed_segsites = np.nanmean(observed_segsites[:, num_burnin:], axis=-1)
mean_observed_tajima_d = np.nanmean(observed_tajima_d[:, num_burnin:], axis=-1)
mean_observed_afs = np.mean(observed_afs[:, num_burnin:], axis=-1)


if snakemake.output.diversity_scatter is not None:
    plt.figure(figsize=(5, 4))
    plt.scatter(mean_observed_diversity, mean_expected_diversity, c="firebrick", s=8)
    plt.axline(
        (np.nanmean(mean_observed_diversity), np.nanmean(mean_observed_diversity)), 
        slope=1, color='black', linestyle="dashed",
    )
    plt.xlabel("Site diversity per window")
    plt.ylabel("E[diversity] per window")
    plt.tight_layout()
    plt.savefig(snakemake.output.diversity_scatter)
    plt.clf()

if snakemake.output.segsites_scatter is not None:
    plt.figure(figsize=(5, 4))
    plt.scatter(mean_observed_segsites, mean_expected_segsites, c="firebrick", s=8)
    plt.axline(
        (np.nanmean(mean_observed_segsites), np.nanmean(mean_observed_segsites)), 
        slope=1, color='black', linestyle="dashed",
    )
    plt.xlabel("Site segsites per window")
    plt.ylabel("E[segregating sites] per window")
    plt.tight_layout()
    plt.savefig(snakemake.output.segsites_scatter)
    plt.clf()

if snakemake.output.tajima_d_scatter is not None:
    plt.figure(figsize=(5, 4))
    plt.scatter(mean_observed_tajima_d, mean_expected_tajima_d, c="firebrick", s=8)
    plt.axline(
        (np.nanmean(mean_observed_tajima_d), np.nanmean(mean_observed_tajima_d)), 
        slope=1, color="black", linestyle="dashed",
    )
    plt.xlabel("Site Tajima's D per window")
    plt.ylabel("E[Tajima's D] per window")
    plt.tight_layout()
    plt.savefig(snakemake.output.tajima_d_scatter)
    plt.clf()

if snakemake.output.diversity_trace is not None:
    plt.figure(figsize=(5, 4))
    plt.plot(mcmc_iterates, trace_expected_diversity, "-", c="firebrick", label="expected")
    plt.plot(mcmc_iterates, trace_observed_diversity, "-", c="black", label="observed")
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[diversity]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.diversity_trace)
    plt.clf()

if snakemake.output.segsites_trace is not None:
    plt.figure(figsize=(5, 4))
    plt.plot(mcmc_iterates, trace_expected_segsites, "-", c="firebrick", label="expected")
    plt.plot(mcmc_iterates, trace_observed_segsites, "-", c="black", label="observed")
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[segregating sites]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.segsites_trace)
    plt.clf()

if snakemake.output.tajima_d_trace is not None:
    plt.figure(figsize=(5, 4))
    plt.plot(mcmc_iterates, trace_expected_tajima_d, "-", c="firebrick", label="expected")
    plt.plot(mcmc_iterates, trace_observed_tajima_d, "-", c="black", label="observed")
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[Tajima's D]")
    plt.legend()
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

if snakemake.output.multimapped_trace is not None:
    plt.figure(figsize=(5, 4))
    plt.plot(mcmc_iterates, multimapped, "-", c="firebrick")
    plt.xlabel("MCMC iteration")
    plt.ylabel("# origins per site")
    plt.tight_layout()
    plt.savefig(snakemake.output.multimapped_trace)
    plt.clf()

if snakemake.output.mutational_load_trace is not None:
    plt.figure(figsize=(5, 4))
    for i, (x, y) in enumerate(zip(observed_load, expected_load)):
        plt.plot(mcmc_iterates, x / y, "-", color="black", linewidth=0.5, alpha=0.25)
    plt.xlabel("MCMC iteration")
    plt.ylabel("(# derived / expectation) per sample")
    plt.tight_layout()
    plt.savefig(snakemake.output.mutational_load_trace)
    plt.clf()

if snakemake.output.diversity_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_expected_diversity[0], quant_expected_diversity[1], color="firebrick", alpha=0.1)
    plt.plot(coord, mean_expected_diversity, "-o", c="firebrick", label="expected", markersize=3)
    plt.plot(coord, mean_observed_diversity, "-o", c="black", label="observed", markersize=3)
    plt.xlabel("Position on chromosome")
    plt.ylabel("Diversity per base")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.diversity_skyline)
    plt.clf()

if snakemake.output.segsites_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_expected_segsites[0], quant_expected_segsites[1], color="firebrick", alpha=0.1)
    plt.plot(coord, mean_expected_segsites, "-o", c="firebrick", label="expected", markersize=3)
    plt.plot(coord, mean_observed_segsites, "-o", c="black", label="observed", markersize=3)
    plt.xlabel("Position on chromosome")
    plt.ylabel("Segregating sites per base")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.segsites_skyline)
    plt.clf()

if snakemake.output.tajima_d_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_expected_tajima_d[0], quant_expected_tajima_d[1], color="firebrick", alpha=0.1)
    plt.plot(coord, mean_expected_tajima_d, "-o", c="firebrick", label="expected", markersize=3)
    plt.plot(coord, mean_observed_tajima_d, "-o", c="black", label="observed", markersize=3)
    plt.xlabel("Position on chromosome")
    plt.ylabel("Tajima's D")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.tajima_d_skyline)
    plt.clf()

if snakemake.output.mutational_load is not None:
    samples = np.arange(mean_expected_load.size)
    plt.figure(figsize=(max(5, 0.1 * samples.size), 4))
    plt.plot(samples, mean_expected_load, "o", color="firebrick", markersize=2)
    plt.vlines(samples, *quant_expected_load, color="firebrick", label="expected")
    plt.plot(samples, mean_observed_load, "o", color="black", markersize=2)
    plt.vlines(samples, *quant_observed_load, color="black", label="observed")
    plt.xlabel("Sample ID")
    plt.ylabel("# derived mutations per base")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.mutational_load)
    plt.clf()

if snakemake.output.frequency_spectrum is not None:
    # TODO: add fold change relative to observed spectrum
    plt.figure(figsize=(8, 4))
    freq = np.arange(1, mean_expected_afs.size)
    plt.fill_between(freq, quant_expected_afs[0][1:], quant_expected_afs[1][1:], color="firebrick", alpha=0.1)
    plt.scatter(freq, mean_expected_afs[1:], c="firebrick", label="expected", s=8)
    plt.scatter(freq, mean_observed_afs[1:], c="black", label="observed", s=8)
    plt.xlabel("Derived allele frequency")
    plt.ylabel("# variants per base")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(snakemake.output.frequency_spectrum)
    plt.clf()


# TODO: clean this up
# stratified summary stats
if snakemake.params.stratify is not None:
    strata = pickle.load(open(snakemake.input.expected_strata_stats[0], "rb"))["strata"]

    # statistics from MCMC samples of trees
    expected_strata_divergence = []
    expected_strata_afs = [[] for _ in strata]
    for i, stats_file in enumerate(snakemake.input.expected_strata_stats):
        expected_strata_stats = pickle.load(open(stats_file, "rb"))
        assert expected_strata_stats["strata"] == strata
        expected_strata_divergence.append(expected_strata_stats["divergence"])
        for j, x in enumerate(expected_strata_stats["afs"]):
            expected_strata_afs[j].append(x)
    expected_strata_divergence = np.stack(expected_strata_divergence, axis=-1)
    expected_strata_afs = [np.stack(x, axis=-1) for x in expected_strata_afs]

    observed_strata_divergence = []
    observed_strata_afs = [[] for _ in strata]
    for i, stats_file in enumerate(snakemake.input.observed_strata_stats):
        observed_strata_stats = pickle.load(open(stats_file, "rb"))
        assert observed_strata_stats["strata"] == strata
        observed_strata_divergence.append(observed_strata_stats["divergence"])
        for j, x in enumerate(observed_strata_stats["afs"]):
            observed_strata_afs[j].append(x)
    observed_strata_divergence = np.stack(observed_strata_divergence, axis=-1)
    observed_strata_afs = [np.stack(x, axis=-1) for x in observed_strata_afs]

    # posterior mean, quantiles
    trace_expected_strata_divergence = np.nanmean(expected_strata_divergence, axis=0)
    quant_expected_strata_divergence = np.nanquantile(
        expected_strata_divergence[..., num_burnin:], 
        quantiles, 
        axis=-1,
    )
    quant_expected_strata_afs = [
        np.quantile(x[..., num_burnin:], quantiles, axis=1) 
        for x in expected_strata_afs
    ]
    mean_expected_strata_divergence = np.nanmean(
        expected_strata_divergence[..., num_burnin:], 
        axis=-1,
    )
    mean_expected_strata_afs = [
        np.mean(x[..., num_burnin:], axis=-1) 
        for x in expected_strata_afs
    ]

    trace_observed_strata_divergence = np.nanmean(observed_strata_divergence, axis=0)
    quant_observed_strata_divergence = np.nanquantile(
        observed_strata_divergence[..., num_burnin:], 
        quantiles, 
        axis=-1,
    )
    quant_observed_strata_afs = [
        np.quantile(x[..., num_burnin:], quantiles, axis=1) 
        for x in observed_strata_afs
    ]
    mean_observed_strata_divergence = np.nanmean(
        observed_strata_divergence[..., num_burnin:], 
        axis=-1,
    )
    mean_observed_strata_afs = [
        np.mean(x[..., num_burnin:], axis=-1) 
        for x in observed_strata_afs
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
                    mean_observed_strata_divergence[:, k], 
                    mean_expected_strata_divergence[:, k], 
                    c="firebrick", s=8,
                )
                x = np.nanmean(mean_observed_strata_divergence[:, k])
                axs[i, j].axline((x, x), slope=1, color="black", linestyle="dashed")
                axs[i, j].set_title(f"{p} vs {q}", loc="left", size=12)
                k += 1
            else:
                axs[i, j].set_visible(False)
    fig.supxlabel("Site divergence per window")
    fig.supylabel("E[divergence] per window")
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
        freq = np.arange(1, mean_expected_strata_afs[i].size)
        axs[i, 0].fill_between(
            freq, 
            quant_expected_strata_afs[i][0, 1:], 
            quant_expected_strata_afs[i][1, 1:], 
            color="firebrick", 
            alpha=0.1,
        )
        axs[i, 0].scatter(
            freq, mean_expected_strata_afs[i][1:], 
            c="firebrick", label="expected", s=8,
        )
        axs[i, 0].scatter(
            freq, mean_observed_strata_afs[i][1:], 
            c="black", label="observed", s=8,
        )
        axs[i, 0].set_title(f"{p}", loc="left", size=12)
        axs[i, 0].set_yscale("log")
        axs[i, 0].legend()
    fig.supxlabel("Derived allele frequency")
    fig.supylabel("# variants per base")
    plt.savefig(strata_afs)
    plt.clf()
    
