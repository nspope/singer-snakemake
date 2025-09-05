"""
Sanity check the time windowed statistics by asserting that the sum over time
windows equals the tskit calculation. A nuance is that multiallelic sites
are handled differently in tskit "site" mode, so these are removed first.
"""

import numpy as np
import msprime
import tskit
import pytest
from workflow.validation.utils import time_windowed_relatedness
from workflow.validation.utils import time_windowed_afs
from workflow.validation.utils import ancestral_state_and_frequency


@pytest.mark.parametrize("populations", [1, 2, 3])
def test_ancestral_state_and_frequency(populations):
    demogr = msprime.Demography.island_model(
        [1e4] * populations, 
        migration_rate=1e-5,
    )
    ts = msprime.sim_ancestry(
        {p.name: 5 for p in demogr.populations}, 
        demography=demogr,
        recombination_rate=1e-8,
        sequence_length=1e5,
        random_seed=1024,
    )
    ts = msprime.sim_mutations(
        ts, 
        rate=1e-8, 
        random_seed=1024,
    )
    sample_sets = [ts.samples(population=i) for i in range(populations)]
    site_state, site_freq = ancestral_state_and_frequency(ts, sample_sets)
    haplotypes = ts.genotype_matrix()
    site_freq_ck = np.stack([np.sum(haplotypes[:, s] == 0, axis=-1) for s in sample_sets], axis=-1)
    site_state_ck = np.array([s.ancestral_state for s in ts.sites()])
    assert np.all(site_freq == site_freq_ck)
    assert np.all(site_state == site_state_ck)


@pytest.mark.parametrize("unknown_age", [True, False])
@pytest.mark.parametrize("populations", [1, 2, 3])
def test_time_windowed_afs(populations, unknown_age):
    demogr = msprime.Demography.island_model(
        [1e4] * populations, 
        migration_rate=1e-5,
    )
    ts = msprime.sim_ancestry(
        {p.name: 5 for p in demogr.populations}, 
        demography=demogr,
        recombination_rate=1e-8,
        sequence_length=1e5,
        random_seed=1024,
    )
    ts = msprime.sim_mutations(
        ts, 
        rate=1e-8, 
        random_seed=1024,
    )
    sample_sets = [list(ts.samples(population=p)) for p in range(populations)]
    breaks = np.append(np.linspace(0, 1e4, 5), np.inf)
    recurrent = np.flatnonzero(np.bincount(ts.mutations_site, minlength=ts.num_sites) != 1)
    ts = ts.delete_sites(recurrent)
    afs = time_windowed_afs(
        ts, 
        sample_sets=sample_sets,
        time_breaks=breaks, 
        unknown_mutation_age=unknown_age, 
    )
    afs_ck = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode='site',
        span_normalise=False,
        polarised=True,
    )
    np.testing.assert_allclose(
        afs.sum(axis=-1), 
        afs_ck,
    )


@pytest.mark.parametrize("unknown_age", [True, False])
@pytest.mark.parametrize("ploidy", [1, 2, 3])
def test_time_windowed_relatedness(ploidy, unknown_age):
    ts = msprime.sim_ancestry(
        10, 
        population_size=1e4,
        ploidy=ploidy,
        recombination_rate=1e-8,
        sequence_length=1e5,
        random_seed=1024,
    )
    ts = msprime.sim_mutations(
        ts, 
        rate=1e-8, 
        random_seed=1024,
    )
    breaks = np.append(np.linspace(0, 1e4, 5), np.inf)
    recurrent = np.flatnonzero(np.bincount(ts.mutations_site, minlength=ts.num_sites) != 1)
    ts = ts.delete_sites(recurrent)
    relatedness = time_windowed_relatedness(
        ts, 
        time_breaks=breaks, 
        unknown_mutation_age=unknown_age, 
        for_individuals=True,
    )
    indices = np.triu_indices(ts.num_individuals)
    relatedness_ck = ts.genetic_relatedness(
        sample_sets=[i.nodes for i in ts.individuals()],
        indexes=list(zip(*indices)),
        mode='site',
        span_normalise=False,
        centre=False,
        proportion=False,
    )
    np.testing.assert_allclose(
        relatedness.sum(axis=-1)[indices], 
        relatedness_ck,
    )


