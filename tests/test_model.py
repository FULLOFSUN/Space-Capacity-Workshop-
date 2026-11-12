"""Sanity tests for the physics building blocks. Run with:  pytest -q"""
import numpy as np
import pytest

from debris2050 import breakup, collisions
from debris2050.atmosphere import circular_lifetime_yr, density
from debris2050.constants import RE
from debris2050.dynamics import disposal_perigee_km, orbit_lifetime_yr


def test_density_decreases_with_altitude():
    h = np.array([200, 400, 600, 800, 1000])
    assert np.all(np.diff(density(h)) < 0)


def test_lifetime_table_matches_circular_integral():
    for h in (400.0, 600.0, 800.0):
        table = orbit_lifetime_yr(np.array([h]), np.array([h]), 0.01)[0]
        exact = circular_lifetime_yr(h, 0.01)
        assert table == pytest.approx(exact, rel=0.01)


def test_disposal_orbit_meets_target_lifetime():
    B = np.array([0.055])
    hp = disposal_perigee_km(np.array([1100.0]), B, 25.0)
    assert orbit_lifetime_yr(hp, np.array([1100.0]), B)[0] <= 25.0
    assert orbit_lifetime_yr(hp + 10, np.array([1100.0]), B)[0] > 25.0


def test_time_fraction_sums_to_one_inside_leo():
    rng = np.random.default_rng(0)
    a = RE + rng.uniform(300e3, 1500e3, 200)
    e = rng.uniform(0, 0.02, 200)
    occ = collisions.occupancy(a, e)
    assert np.allclose(np.bincount(occ.obj, occ.w, a.size), 1.0)


def test_sparse_rates_equal_brute_force():
    rng = np.random.default_rng(1)
    n = 150
    a = RE + rng.uniform(400e3, 1200e3, n)
    e = rng.uniform(0, 0.03, n)
    r = rng.uniform(1e-4, 3e-3, n)
    occ = collisions.occupancy(a, e)
    f = np.zeros((n, collisions.N_SHELLS))
    np.add.at(f, (occ.obj, occ.shell), occ.w)
    brute = np.zeros(collisions.N_SHELLS)
    for k in range(collisions.N_SHELLS):
        w = f[:, k]
        s = w[:, None] * w[None, :] * (r[:, None] + r[None, :]) ** 2
        np.fill_diagonal(s, 0)
        brute[k] = collisions._RATE_COEF[k] * s.sum() / 2
    assert np.allclose(collisions.shell_rates(occ, r), brute, rtol=1e-10, atol=0)
    assert collisions.object_rates(occ, r).sum() / 2 == pytest.approx(brute.sum())


def test_catastrophic_threshold():
    # EMR = 0.5 * m_p * v^2 / m_t; 40 J/g at 10 km/s needs m_p/m_t >= 8e-4
    assert breakup.is_catastrophic(1000.0, 1.0)
    assert not breakup.is_catastrophic(1000.0, 0.5)


def test_fragment_cloud_respects_mass_budget():
    rng = np.random.default_rng(3)
    frags = breakup.fragment_cloud(800.0, 600.0, 20.0, True, RE + 550e3, 53.0, rng)
    assert frags["a"].size > 50
    assert frags["mass"].sum() <= 1400.0
    assert np.all(frags["a"] * (1 - frags["e"]) > RE)
