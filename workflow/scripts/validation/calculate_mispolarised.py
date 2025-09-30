"""
Calculate the proportion of sites that are correctly/incorrectly repolarised
(ancestral state switched relative to input), as a function of true frequency
(AFS bin).  Optionally, AFS bins are down-projected to lower frequencies via
hypergeometric sampling, which can smooth results for smaller sequence lengths.

With regards to the true ancestral state, the first two dimensions of the output
array corresponds to the number of sites that are:

    (0, 0) Correct in both VCF and SINGER trees
    (0, 1) Correct in VCF and incorrect in SINGER trees
    (1, 0) Incorrect in VCF and correct in SINGER trees
    (1, 1) Incorrect in both VCF and SINGER trees

The remaining dimensions index the AFS bin. Counts are averaged over MCMC
samples.

Part of https://github.com/nspope/singer-snakemake.
"""

import numpy as np
import pickle
import tszip

from string import ascii_lowercase, ascii_uppercase
from utils import hypergeometric_probabilities
from utils import ancestral_state_and_frequency


project_to = snakemake.params.project_to
position_adjust = snakemake.params.position_adjust

treefiles = iter(snakemake.input.infr_trees)
vcf_alleles = pickle.load(open(snakemake.input.vcf_alleles, "rb"))
true_ts = tszip.load(snakemake.input.true_trees)
infr_ts = tszip.load(next(treefiles))
sample_populations = np.unique([true_ts.nodes_population[i] for i in true_ts.samples()])
sample_sets = [true_ts.samples(population=p) for p in sample_populations]
afs_dim = [len(s) + 1 for s in sample_sets]

# remove mutations that are (actually) recurrent
recurrent = np.bincount(true_ts.mutations_site, minlength=true_ts.num_sites) != 1
true_ts = true_ts.delete_sites(recurrent)

# extract true ancestral state and frequency
ancestral_state, ancestral_frequency = ancestral_state_and_frequency(true_ts, sample_sets)
true_position = true_ts.sites_position + position_adjust
true_ancestral = {p: a for p, a in zip(true_position, ancestral_state)}
true_frequency = {p: f for p, f in zip(true_position, ancestral_frequency)}
vcf_ancestral = {p: x[0] for p, x in vcf_alleles.items()}

def mispolarisations_by_true_frequency(infr_ts):
    """
    Count number of sites that are correctly polarised in the input
    and after being run through SINGER, as a function of (true)
    frequency of ancestral state.
    """
    counts = np.zeros([2, 2] + afs_dim)
    for s in infr_ts.sites():
        p = s.position
        if p in true_frequency:
            f = true_frequency[p].astype(int)
            anc, inp = true_ancestral[p], vcf_ancestral[p]
            inf = s.ancestral_state
            counts[0, 0][*f] += (inp == anc) and (inf == anc)
            counts[0, 1][*f] += (inp == anc) and (inf != anc)
            counts[1, 0][*f] += (inp != anc) and (inf == anc)
            counts[1, 1][*f] += (inp != anc) and (inf != anc)
    return counts

mispolarised = mispolarisations_by_true_frequency(infr_ts)
for infr_trees in treefiles:
    mispolarised += mispolarisations_by_true_frequency(tszip.load(infr_trees))
mispolarised /= len(snakemake.input.infr_trees)

if project_to is not None:  # down-project mutation frequencies
    projection = [hypergeometric_probabilities(len(s), project_to) for s in sample_sets]
    dim = len(projection)
    # einsum string has the form 'uvijk,iI,jJ,kK->uvIJK'
    lhs = []
    lhs.append('uv' + ''.join([x for x in ascii_lowercase[:dim]]))
    lhs.extend([x + y for x, y in zip(ascii_lowercase[:dim], ascii_uppercase[:dim])])
    lhs = ','.join(lhs)
    rhs = 'uv' + ''.join([x for x in ascii_uppercase[:dim]])
    mispolarised = np.einsum(f"{lhs}->{rhs}", mispolarised, *projection)

np.save(snakemake.output.mispolarised, mispolarised)

