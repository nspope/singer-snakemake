import msprime
import numpy as np
import tskit
import tszip
import gzip
import os
import csv
import demes
import argparse

docstring = \
"""
Simulate a pipeline example: four genes (exons under a 4D-like mask), selection
interleaved (neutral / swept), masking split by position — first two genes use
the dating mask (SNPs kept for tree building, dropped for dating), last two use
a hard accessibility mask (SNPs removed entirely). Masked sites are simulated
neutrally: this probes data sparsity / genealogy dependence, not mutational-
process variation or codon selection. Writes pipeline inputs, dating.mask.bed,
mask.bed, and the true trees.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--sequence_length", default=1e6, type=float)
parser.add_argument("--population_size", default=1e4, type=float)
parser.add_argument("--mutation_rate", default=1.25e-8, type=float)
parser.add_argument("--recombination_rate", default=1.25e-8, type=float)
parser.add_argument("--samples", default=30, type=int)
parser.add_argument("--sweep_s", default=0.05, type=float)
parser.add_argument("--num_exons", default=5, type=int)
parser.add_argument("--exon_length", default=5000, type=int)
parser.add_argument("--intron_length", default=5000, type=int)
parser.add_argument("--haploid", action="store_true")
parser.add_argument("--disable_repolarise", action="store_true")
parser.add_argument("--output_prefix", default="example/genic/genic", type=str)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
ploidy = 1 if args.haploid else 2
L = int(args.sequence_length)
Ne = args.population_size

dir_name = os.path.dirname(args.output_prefix)
if not os.path.exists(dir_name): os.makedirs(dir_name)


# gene = num_exons exons separated by introns (length 0 -> abutting exons)
def gene_exons(center):
    span = args.num_exons * args.exon_length + (args.num_exons - 1) * args.intron_length
    p = int(center - span / 2)
    exons = []
    for _ in range(args.num_exons):
        exons.append((p, p + args.exon_length))
        p += args.exon_length + args.intron_length
    return exons

# four genes; selection interleaved; masking split by position
gene_centers = [int(f * L) for f in (0.2, 0.4, 0.6, 0.8)]
gene_exon_sets = [gene_exons(c) for c in gene_centers]
sweep_genes = [1, 3]    # interleaved with neutral genes 0, 2
dating_genes = [0, 1]   # non-4D kept in VCF, dropped for dating
hard_genes = [2, 3]     # non-4D removed entirely

# sweep at the centre of each swept gene's central exon
def central_exon_centre(exset):
    mid = exset[len(exset) // 2]
    return int((mid[0] + mid[1]) / 2)
sweep_positions = [central_exon_centre(gene_exon_sets[i]) for i in sweep_genes]


# constant-size single population
demo = msprime.Demography()
demo.add_population(name="A", initial_size=Ne)
samples = [msprime.SampleSet(num_samples=args.samples, population="A", ploidy=ploidy)]


# one sweep per swept gene, chained (each completes before the next, so they
# occur at slightly different times), then the standard coalescent
sweep_models = [
    msprime.SweepGenicSelection(
        position=p,
        start_frequency=1.0 / (2 * Ne),
        end_frequency=1.0 - 1.0 / (2 * Ne),
        s=args.sweep_s,
        dt=1.0 / (40 * Ne),
    )
    for p in sweep_positions
]
ts = msprime.sim_ancestry(
    samples=samples,
    demography=demo,
    sequence_length=L,
    recombination_rate=args.recombination_rate,
    model=[*sweep_models, msprime.StandardCoalescent()],
    random_seed=args.seed,
)
ts = msprime.sim_mutations(ts, rate=args.mutation_rate, random_seed=args.seed)


# per exon keep every 3rd base (4D-like), mask the other 2/3
def comb_bitmask(gene_indices):
    bm = np.full(L, False)
    for gi in gene_indices:
        for a, b in gene_exon_sets[gi]:
            idx = np.arange(a, b)
            bm[idx[(idx - a) % 3 != 2]] = True
    return bm
dating_bitmask = comb_bitmask(dating_genes)
hard_bitmask = comb_bitmask(hard_genes)


# ancestral sequence (true states, pre-repolarisation), N elsewhere and over the hard mask
ancestral_sequence = np.full(L, "N")
for s in ts.sites():
    ancestral_sequence[int(s.position)] = s.ancestral_state
ancestral_sequence[hard_bitmask] = "N"


# save trees for reference (full ARG, before hard-mask site removal)
tszip.compress(ts, f"{args.output_prefix}.tsz")


# hard mask: remove those SNPs entirely (excluded from tree building AND dating)
hard_sites = np.flatnonzero(hard_bitmask[ts.sites_position.astype(np.int64)])
ts = ts.delete_sites(hard_sites)


# randomly choose ref/alt state
if not args.disable_repolarise:
    repolarise = rng.binomial(1, 0.5, size=ts.num_sites).astype(bool)
    tab = ts.dump_tables()
    tab.mutations.clear()
    tab.sites.clear()
    tree = ts.first()
    for site in ts.sites():
        tree.seek(site.position)
        biallelic = len(site.mutations) == 1
        if repolarise[site.id] and biallelic:
            mutation = next(iter(site.mutations))
            for r in tree.roots:
                tab.mutations.add_row(site=site.id, node=r, derived_state=site.ancestral_state)
            tab.sites.add_row(position=site.position, ancestral_state=mutation.derived_state)
        else:
            tab.sites.add_row(position=site.position, ancestral_state=site.ancestral_state)
        for mutation in site.mutations:
            tab.mutations.add_row(site=mutation.site, node=mutation.node, derived_state=mutation.derived_state)
    tab.sort()
    tab.build_index()
    tab.compute_mutation_parents()
    assert tab.sites.num_rows == ts.num_sites
    assert np.all(tab.sites.position == ts.sites_position)
    ts = tab.tree_sequence()


# write out VCF
individual_names = [f"Sample{i:03d}" for i in range(args.samples)]
ts.write_vcf(
    gzip.open(f"{args.output_prefix}.vcf.gz", "wt"),
    position_transform=lambda x: np.array(x, dtype=np.int64) + 1,
    individual_names=individual_names,
)


# write out flat recombination map
recmap = msprime.RateMap(position=np.linspace(0, L, 1001), rate=np.full(1000, args.recombination_rate))
hapmap_path = f"{args.output_prefix}.hapmap"
if os.path.exists(hapmap_path): os.remove(hapmap_path)
with open(hapmap_path, "w") as hapmap:
    hapmap.write("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n")
    for start, rate, mappos in zip(
            recmap.position[1:],
            np.append(1e8 * recmap.rate[1:], 0.0),
            recmap.get_cumulative_mass(recmap.position[1:]) * 100,
        ):
            hapmap.write(f"1\t{int(start)}\t{rate:.8f}\t{mappos:.8f}\n")
hapmap_check = msprime.RateMap.read_hapmap(hapmap_path, rate_col=2)
np.testing.assert_allclose(hapmap_check.rate[1:], recmap.rate[1:], atol=1e-12)
np.testing.assert_allclose(hapmap_check.position, recmap.position)


# write out metadata
meta_path = f"{args.output_prefix}.meta.csv"
if os.path.exists(meta_path): os.remove(meta_path)
with open(meta_path, "w") as metafile:
    metafile.write("name,population\n")
    for samp in individual_names:
        metafile.write(f"{samp},\"A\"\n")
meta_check = csv.reader(open(meta_path, "r"))
colnames = next(meta_check)
assert colnames[0] == 'name' and colnames[1] == 'population'
for i, row in enumerate(meta_check):
    assert row[0] == individual_names[i]
    assert row[1] == "A"


# write out demography
demes.dump(demo.to_demes(), f"{args.output_prefix}.demes.yaml")


# write out ancestral sequence
ancestral_path = f"{args.output_prefix}.ancestral.fa.gz"
if os.path.exists(ancestral_path): os.remove(ancestral_path)
with gzip.open(ancestral_path, "wt") as fastafile:
    fastafile.write(">1\n")
    fastafile.write("".join(ancestral_sequence) + "\n")


# write out masks as 0-based half-open BED
def write_bed(bitmask, path):
    if os.path.exists(path): os.remove(path)
    padded = np.concatenate([[False], bitmask, [False]]).astype(np.int8)
    diff = np.diff(padded)
    intervals = np.column_stack([np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)])
    with open(path, "w") as bedfile:
        for a, b in intervals:
            bedfile.write(f"1\t{int(a)}\t{int(b)}\n")
    check = np.loadtxt(path, usecols=[1, 2]).reshape(-1, 2)
    np.testing.assert_allclose(intervals, check)

# dating mask (kept in VCF, dropped for dating); hard mask (removed entirely)
write_bed(dating_bitmask, f"{args.output_prefix}.dating.mask.bed")
write_bed(hard_bitmask, f"{args.output_prefix}.mask.bed")
