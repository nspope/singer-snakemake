import msprime
import numpy as np
import tszip
import gzip
import os
import csv
import demes
import argparse

docstring = \
"""
Simulate example inputs for pipeline, with missing data and variable recombination rates.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--sequence_length", default=10e6, type=float)
parser.add_argument("--population_size", default=1e4, type=float)
parser.add_argument("--migration_rate", default=1e-5, type=float)
parser.add_argument("--num_populations", default=3, type=int)
parser.add_argument("--mutation_rate", default=1e-8, type=float)
parser.add_argument("--samples", default=10, type=int)
parser.add_argument("--disable-mask", action="store_true")
parser.add_argument("--disable-hapmap", action="store_true")
parser.add_argument("--disable-meta", action="store_true")
parser.add_argument("--disable-filter", action="store_true")
parser.add_argument("--haploid", action="store_true")
parser.add_argument("--output-prefix", default="example/example", type=str)
parser.add_argument("--remove-missing", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

dir_name = os.path.dirname(args.output_prefix)
if not os.path.exists(dir_name): os.makedirs(dir_name)

# linearly increasing recombination rate
recmap_pos = np.linspace(0, args.sequence_length, 1001)
recmap_rates = rng.uniform(0, args.mutation_rate * (0.01 + recmap_pos[:-1] / recmap_pos[-1]))
if args.disable_hapmap: recmap_rates[:] = args.mutation_rate
recmap = msprime.RateMap(position=recmap_pos, rate=recmap_rates)

demo = msprime.Demography.island_model(
    initial_size=[args.population_size for _ in range(args.num_populations)], 
    migration_rate=args.migration_rate,
)
samples = [
    msprime.SampleSet(num_samples=args.samples, population=pop.name, ploidy=1 if args.haploid else 2) 
    for pop in demo.populations
]
ts = msprime.sim_ancestry(
    samples=samples,
    recombination_rate=recmap,
    demography=demo,
    random_seed=args.seed,
)
ts = msprime.sim_mutations(ts, rate=args.mutation_rate, random_seed=args.seed)


# sporadic missing data with large missing segment in center
bedmask = []
sequence_length = ts.sequence_length
missing_start = int(0.4 * ts.sequence_length)
missing_end = int(0.6 * ts.sequence_length)
end = 0
while missing_start - end > 1e3:
    start = rng.integers(end, min(end + 1e3, missing_start))
    end = rng.integers(start, min(start + 1e3, missing_end))
    bedmask.append([start, end])
bedmask.append([missing_start, missing_end])
end = missing_end
while ts.sequence_length - end > 1e3:
    start = rng.integers(end, min(end + 1e3, ts.sequence_length))
    end = rng.integers(start, min(start + 1e3, ts.sequence_length))
    bedmask.append([start, end])
bedmask = np.array(bedmask)
# add some large gaps on top of the sporadic missingness
gaps = np.array([
    [0.03, 0.08], 
    [0.14, 0.19], 
    [0.25, 0.30], 
    [0.36, 0.41], 
    [0.47, 0.52],
]) * ts.sequence_length
bedmask = np.concatenate([bedmask, gaps], axis=0).astype(int)
# convert to bitmask
bitmask = np.full(int(recmap.sequence_length) + 1, False)
for a, b in bedmask: bitmask[a:b] = True


# sporadic filtered snps in latter half of chunks
remaining_sites = ts.sites_position[~bitmask[ts.sites_position.astype(np.int64)]]
remaining_sites = remaining_sites[remaining_sites > ts.sequence_length / 2]
sitemask = np.sort(
    rng.choice(
        remaining_sites.astype(np.int64) + 1,
        size=int(0.3 * remaining_sites.size),
        replace=False,
    )
)


# actually remove masked variants from VCF
if args.remove_missing:
    snpmask = np.full(int(recmap.sequence_length) + 1, False)
    if not args.disable_filter:
        snpmask[sitemask - 1] = True
    if not args.disable_mask:
        snpmask = np.logical_or(snpmask, bitmask)
    remove_sites = np.flatnonzero(snpmask[ts.sites_position.astype(np.int64)])
    ts = ts.delete_sites(remove_sites)


# write out VCF
population_names = np.concatenate([np.repeat(s.population, s.num_samples) for s in samples])
individual_names = [f"Sample{i:03d}" for i in range(population_names.size)]
ts.write_vcf(
    gzip.open(f"{args.output_prefix}.vcf.gz", "wt"), 
    position_transform=lambda x: np.array(x, dtype=np.int64) + 1, 
    individual_names=individual_names,
)
tszip.compress(ts, f"{args.output_prefix}.tsz")

hapmap_path = f"{args.output_prefix}.hapmap"
if os.path.exists(hapmap_path): os.remove(hapmap_path)
if not args.disable_hapmap:
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
    hapmap_check = msprime.RateMap.read_hapmap(hapmap_path, map_col=3)
    np.testing.assert_allclose(hapmap_check.rate, recmap.rate, atol=1e-12)
    np.testing.assert_allclose(hapmap_check.position, recmap.position)

mask_path = f"{args.output_prefix}.mask.bed"
if os.path.exists(mask_path): os.remove(mask_path)
if not args.disable_mask:
    with open(mask_path, "w") as maskfile:
        for a, b in bedmask:
            maskfile.write(f"1\t{int(a)}\t{int(b)}\n")
    bedmask_check = np.loadtxt(mask_path, usecols=[1,2])
    np.testing.assert_allclose(bedmask, bedmask_check)

meta_path = f"{args.output_prefix}.meta.csv"
if os.path.exists(meta_path): os.remove(meta_path)
if not args.disable_meta:
    with open(meta_path, "w") as metafile:
        metafile.write("name,population\n")
        for samp, pop in zip(individual_names, population_names):
            metafile.write(f"{samp},\"{pop}\"\n")
    meta_check = csv.reader(open(meta_path, "r"))
    colnames = next(meta_check)
    assert colnames[0] == 'name' and colnames[1] == 'population'
    for i, row in enumerate(meta_check):
        assert row[0] == individual_names[i]
        assert row[1] == population_names[i]

filter_path = f"{args.output_prefix}.filter.txt"
if os.path.exists(filter_path): os.remove(filter_path)
if not args.disable_filter:
    with open(filter_path, "w") as filterfile:
        for pos in sitemask: 
            filterfile.write(f"{int(pos)}\n")
    filter_check = np.loadtxt(filter_path)
    assert np.allclose(filter_check, sitemask)

demog_path = f"{args.output_prefix}.demes.yaml"
demes.dump(demo.to_demes(), demog_path)
