import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import msprime

parser = argparse.ArgumentParser()
parser.add_argument("--ratemap", type=str, required=True)
parser.add_argument("--vcf-stats", type=str, required=True)
parser.add_argument("--tree-stats", type=str, nargs="+", required=True)
parser.add_argument("--diversity-scatter", type=str)
parser.add_argument("--diversity-trace", type=str)
parser.add_argument("--diversity-skyline", type=str)
parser.add_argument("--tajima-d-scatter", type=str)
parser.add_argument("--tajima-d-trace", type=str)
parser.add_argument("--tajima-d-skyline", type=str)
parser.add_argument("--folded-afs", type=str)
parser.add_argument("--unfolded-afs", type=str)
parser.add_argument("--conf-level", type=float, default=0.0)
parser.add_argument("--mcmc-burnin", type=float, default=0.2)
args = parser.parse_args()

# statistics from VCF
ratemap = pickle.load(open(args.ratemap, "rb"))
windows = ratemap.position
coord = windows[:-1] / 2 + windows[1:] / 2 
vcf_stats = pickle.load(open(args.vcf_stats, "rb"))
site_diversity = vcf_stats['diversity']
site_tajima_d = vcf_stats['tajima_d']
site_folded_afs = vcf_stats['folded_afs']
site_unfolded_afs = vcf_stats['unfolded_afs']

# statistics from MCMC samples of trees
MCMC_SAMPLES = len(args.tree_stats)
MCMC_BURNIN = int(MCMC_SAMPLES * args.mcmc_burnin)
branch_diversity = np.zeros((windows.size - 1, MCMC_SAMPLES))
branch_tajima_d = np.zeros((windows.size - 1, MCMC_SAMPLES))
branch_folded_afs = np.zeros((site_folded_afs.size, MCMC_SAMPLES))
branch_unfolded_afs = np.zeros((site_unfolded_afs.size, MCMC_SAMPLES))
for i, stats_file in enumerate(args.tree_stats):
    stats = pickle.load(open(stats_file, "rb"))
    branch_diversity[:, i] = stats['diversity']
    branch_tajima_d[:, i] = stats['tajima_d']
    branch_folded_afs[:, i] = stats['folded_afs'][:site_folded_afs.size]
    branch_unfolded_afs[:, i] = stats['unfolded_afs'][:site_unfolded_afs.size]

# posterior mean, quantiles
CONF_LEVEL = args.conf_level
quantiles = np.array([CONF_LEVEL / 2, 1 - CONF_LEVEL / 2])
trace_diversity = np.nanmean(branch_diversity, axis=0)
trace_tajima_d = np.nanmean(branch_tajima_d, axis=0)
quant_diversity = np.nanquantile(branch_diversity[:, MCMC_BURNIN:], quantiles, axis=1)
quant_tajima_d = np.nanquantile(branch_tajima_d[:, MCMC_BURNIN:], quantiles, axis=1)
quant_folded_afs = np.quantile(branch_folded_afs[:, MCMC_BURNIN:], quantiles, axis=1)
quant_unfolded_afs = np.quantile(branch_unfolded_afs[:, MCMC_BURNIN:], quantiles, axis=1)
mean_diversity = np.nanmean(branch_diversity[:, MCMC_BURNIN:], axis=1)
mean_tajima_d = np.nanmean(branch_tajima_d[:, MCMC_BURNIN:], axis=1)
mean_folded_afs = np.mean(branch_folded_afs[:, MCMC_BURNIN:], axis=1)
mean_unfolded_afs = np.mean(branch_unfolded_afs[:, MCMC_BURNIN:], axis=1)

if args.diversity_scatter is not None:
    plt.scatter(site_diversity, mean_diversity, c='firebrick', s=8)
    plt.axline((np.nanmean(site_diversity), np.nanmean(site_diversity)), slope=1, color='black')
    plt.xlabel("Site (VCF) diversity per block")
    plt.ylabel("E[diversity] per block")
    plt.savefig(args.diversity_scatter)
    plt.clf()

if args.tajima_d_scatter is not None:
    plt.scatter(site_tajima_d, mean_tajima_d, c='firebrick', s=8)
    plt.axline((np.nanmean(site_tajima_d), np.nanmean(site_tajima_d)), slope=1, color='black')
    plt.xlabel("Site (VCF) Tajima's D per block")
    plt.ylabel("E[Tajima's D] per block")
    plt.savefig(args.tajima_d_scatter)
    plt.clf()

if args.diversity_trace is not None:
    plt.plot(np.arange(trace_diversity.size), trace_diversity, "-", c='firebrick')
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[diversity]")
    plt.savefig(args.diversity_trace)
    plt.clf()

if args.tajima_d_trace is not None:
    plt.plot(np.arange(trace_tajima_d.size), trace_tajima_d, "-", c='firebrick')
    plt.xlabel("MCMC iteration")
    plt.ylabel("E[Tajima's D]")
    plt.savefig(args.tajima_d_trace)
    plt.clf()

if args.diversity_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_diversity[0], quant_diversity[1], color='firebrick', alpha=0.1)
    plt.plot(coord, mean_diversity, "-", c='firebrick', label='branch-ARG')
    plt.plot(coord, site_diversity, "-", c='black', label='site-VCF')
    plt.xlabel("Position on chromosome")
    plt.ylabel("Diversity / base")
    plt.legend()
    plt.savefig(args.diversity_skyline)
    plt.clf()

if args.tajima_d_skyline is not None:
    plt.figure(figsize=(8, 4))
    plt.fill_between(coord, quant_tajima_d[0], quant_tajima_d[1], color='firebrick', alpha=0.1)
    plt.plot(coord, mean_tajima_d, "-", c='firebrick', label='branch-ARG')
    plt.plot(coord, site_tajima_d, "-", c='black', label='site-VCF')
    plt.xlabel("Position on chromosome")
    plt.ylabel("Tajima's D")
    plt.legend()
    plt.savefig(args.tajima_d_skyline)
    plt.clf()

if args.folded_afs is not None:
    freq = np.arange(1, site_folded_afs.size)
    plt.fill_between(freq, quant_folded_afs[0][1:], quant_folded_afs[1][1:], color='firebrick', alpha=0.1)
    plt.scatter(freq, mean_folded_afs[1:], c='firebrick', label='branch-ARG', s=8)
    plt.scatter(freq, site_folded_afs[1:], c='black', label='site-VCF', s=8)
    plt.xlabel("Minor allele frequency")
    plt.ylabel("# of variants / base")
    plt.yscale("log")
    plt.legend()
    plt.savefig(args.folded_afs)
    plt.clf()

if args.unfolded_afs is not None:
    freq = np.arange(1, site_unfolded_afs.size - 1)
    plt.fill_between(freq, quant_unfolded_afs[0][1:-1], quant_unfolded_afs[1][1:-1], color='firebrick', alpha=0.1)
    plt.scatter(freq, mean_unfolded_afs[1:-1], c='firebrick', label='branch-ARG', s=8)
    plt.scatter(freq, site_unfolded_afs[1:-1], c='black', label='site-VCF', s=8)
    plt.xlabel("Derive allele frequency")
    plt.ylabel("# of variants / base")
    plt.yscale("log")
    plt.legend()
    plt.savefig(args.unfolded_afs)
    plt.clf()
