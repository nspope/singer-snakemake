import msprime
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--sequence_length", default=10e6, type=float)
parser.add_argument("--population_size", default=1e4, type=float)
parser.add_argument("--migration_rate", default=1e-5, type=float)
parser.add_argument("--num_populations", default=2, type=int)
parser.add_argument("--mutation_rate", default=1e-8, type=float)
parser.add_argument("--samples", default=12, type=int)
args = parser.parse_args()

np.random.seed(args.seed)

# linearly increasing recombination rate
recmap_pos = np.linspace(0, args.sequence_length, 1001)
recmap_rates = np.random.uniform(0, args.mutation_rate * (0.01 + recmap_pos[:-1] / recmap_pos[-1]))
recmap = msprime.RateMap(position=recmap_pos, rate=recmap_rates)

demo = msprime.Demography.island_model(
    initial_size=[args.population_size for _ in range(args.num_populations)], 
    migration_rate=args.migration_rate,
)
samples = [msprime.SampleSet(num_samples=args.samples, population=pop.name) for pop in demo.populations]
ts = msprime.sim_ancestry(
    samples=samples,
    recombination_rate=recmap, 
    demography=demo,
    random_seed=args.seed
)
ts = msprime.sim_mutations(ts, rate=args.mutation_rate, random_seed=args.seed)

# linearly increasing missing data
bedmask = []
sequence_length = ts.sequence_length
end = 0
while ts.sequence_length - end > 1e3:
    start = np.random.randint(end, min(end + 1e3, ts.sequence_length))
    end = np.random.randint(start, min(start + 1e3, ts.sequence_length))
    bedmask.append([start, end])
bedmask = np.array(bedmask)
bitmask = np.full(int(recmap.sequence_length) + 1, False)
for a, b in bedmask: bitmask[a:b] = True

mask_sites = np.flatnonzero(bitmask[ts.sites_position.astype(np.int64)])
ts = ts.delete_sites(mask_sites)
population_names = np.concatenate([np.repeat(s.population, s.num_samples) for s in samples])
individual_names = [f"Sample{i:03d}" for i in range(population_names.size)]
ts.write_vcf(open("example.vcf", "w"), position_transform=lambda x: np.array(x, dtype=np.int64) + 1, individual_names=individual_names)

with open("example.hapmap", "w") as hapmap:
    hapmap.write("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n")
    for start, rate, mappos in zip(
            recmap.position[1:],
            np.append(1e8 * recmap.rate[1:], 0.0),
            recmap.get_cumulative_mass(recmap.position[1:]) * 100,
        ):
            hapmap.write(f"1\t{int(start)}\t{rate:.8f}\t{mappos:.8f}\n")
hapmap_check = msprime.RateMap.read_hapmap("example.hapmap", rate_col=2)
np.testing.assert_allclose(hapmap_check.rate[1:], recmap.rate[1:], atol=1e-12)
np.testing.assert_allclose(hapmap_check.position, recmap.position)
hapmap_check = msprime.RateMap.read_hapmap("example.hapmap", map_col=3)
np.testing.assert_allclose(hapmap_check.rate, recmap.rate, atol=1e-12)
np.testing.assert_allclose(hapmap_check.position, recmap.position)

with open("example.mask.bed", "w") as maskfile:
    for a, b in bedmask:
        maskfile.write(f"1\t{int(a)}\t{int(b)}\n")
bedmask_check = np.loadtxt("example.mask.bed", usecols=[1,2])
np.testing.assert_allclose(bedmask, bedmask_check)

with open("example.meta.csv", "w") as metafile:
    metafile.write("name,population\n")
    for samp, pop in zip(individual_names, population_names):
        metafile.write(f"{samp},\"{pop}\"\n")
meta_check = csv.reader(open("example.meta.csv", "r"))
colnames = next(meta_check)
assert colnames[0] == 'name' and colnames[1] == 'population'
for i, row in enumerate(meta_check):
    assert row[0] == individual_names[i]
    assert row[1] == population_names[i]
