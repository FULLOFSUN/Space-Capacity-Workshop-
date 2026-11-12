"""Orbit decay under drag (King-Hele orbit-averaged equations) and an
eccentric-orbit lifetime table used to plan post-mission disposal."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ive  # exponentially scaled modified Bessel I_n(x) * exp(-x)

from .atmosphere import density, scale_height_km
from .constants import (CD, MU, RE, REENTRY_MEAN_ALT_KM, REENTRY_PERIGEE_KM,
                        SEC_PER_YEAR)


def decay_rates(a: np.ndarray, e: np.ndarray, B: np.ndarray,
                rho_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Orbit-averaged da/dt [m/s] and de/dt [1/s] for ballistic coefficient B = Cd A/m.

    da/dt = -B rho_p sqrt(mu a) e^-c [I0(c) + 2e I1(c)]
    de/dt = -B rho_p sqrt(mu/a) e^-c [I1(c) + e/2 (I0(c) + I2(c))]
    with c = a e / H and rho_p the density at perigee
    (King-Hele, *Satellite Orbits in an Atmosphere*, 1987).
    """
    hp_km = (a * (1.0 - e) - RE) / 1e3
    rho_p = density(hp_km) * rho_factor
    c = a * e / (scale_height_km(hp_km) * 1e3)
    i0, i1, i2 = ive(0, c), ive(1, c), ive(2, c)
    dadt = -B * rho_p * np.sqrt(MU * a) * (i0 + 2.0 * e * i1)
    dedt = -B * rho_p * np.sqrt(MU / a) * (i1 + 0.5 * e * (i0 + i2))
    return dadt, dedt


def decay_step(a: np.ndarray, e: np.ndarray, area: np.ndarray, mass: np.ndarray,
               dt_yr: float, rho_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance (a, e) by dt under drag (explicit Euler)."""
    dadt, dedt = decay_rates(a, e, CD * area / mass, rho_factor)
    dt = dt_yr * SEC_PER_YEAR
    return np.maximum(a + dadt * dt, RE), np.clip(e + dedt * dt, 0.0, None)


# ---------------------------------------------------------------------------
# Lifetime table. Because da/dt and de/dt are both proportional to B, the
# lifetime of an orbit scales exactly as 1/B under a fixed atmosphere, so a
# single table L1(hp, ha) computed for B = 1 m^2/kg covers every object:
#     lifetime(hp, ha, B) = L1(hp, ha) / B
# ---------------------------------------------------------------------------
HP_GRID_KM = np.arange(REENTRY_PERIGEE_KM + 30.0, 2001.0, 10.0)
HA_GRID_KM = np.arange(REENTRY_PERIGEE_KM + 30.0, 4001.0, 25.0)


STEP_M = 2.5e3
_CACHE = Path(__file__).with_name("_lifetime_table_b1.npz")


@lru_cache(maxsize=1)
def lifetime_table_b1() -> np.ndarray:
    """L1[i, j]: lifetime [yr * m^2/kg] for perigee HP_GRID[i], apogee HA_GRID[j].

    Built once (~20 s) and cached next to this module."""
    try:
        z = np.load(_CACHE)
        if np.array_equal(z["hp"], HP_GRID_KM) and np.array_equal(z["ha"], HA_GRID_KM):
            return z["L"]
    except (OSError, KeyError, ValueError):
        pass
    L = _build_table()
    try:
        np.savez_compressed(_CACHE, hp=HP_GRID_KM, ha=HA_GRID_KM, L=L)
    except OSError:
        pass
    return L


def _build_table() -> np.ndarray:
    hp, ha = np.meshgrid(HP_GRID_KM, HA_GRID_KM, indexing="ij")
    valid = ha >= hp
    a = RE + 0.5 * (hp + ha) * 1e3
    e = (ha - hp) * 1e3 / (2 * a)
    a, e = a[valid], e[valid]
    t = np.zeros(a.size)
    live = np.ones(a.size, bool)
    for _ in range(20000):
        if not live.any():
            break
        idx = np.flatnonzero(live)
        ones = np.ones(idx.size)
        a0, e0 = a[idx], e[idx]
        dadt, dedt = decay_rates(a0, e0, ones)
        dt = STEP_M / np.abs(dadt)                   # fixed steps in semi-major axis
        am = a0 + 0.5 * dadt * dt                    # midpoint (RK2) step
        em = np.clip(e0 + 0.5 * dedt * dt, 0.0, None)
        dadt_m, dedt_m = decay_rates(am, em, ones)
        dt = STEP_M / np.abs(dadt_m)
        a[idx] = a0 + dadt_m * dt
        e[idx] = np.clip(e0 + dedt_m * dt, 0.0, None)
        t[idx] += dt
        down = ((a[idx] * (1 - e[idx]) - RE) / 1e3 < REENTRY_PERIGEE_KM) | \
               ((a[idx] - RE) / 1e3 < REENTRY_MEAN_ALT_KM)
        live[idx[down]] = False
    L = np.full(hp.shape, np.nan)
    L[valid] = t / SEC_PER_YEAR
    # cells with ha < hp are not orbits; give them the row's circular value so
    # bilinear interpolation next to the diagonal stays finite
    for i in range(L.shape[0]):
        first = np.flatnonzero(~np.isnan(L[i]))
        if first.size:
            L[i, :first[0]] = L[i, first[0]]
    return L


def orbit_lifetime_yr(hp_km: np.ndarray, ha_km: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Natural lifetime [yr] under the mean atmosphere (bilinear table look-up)."""
    L = lifetime_table_b1()
    hp = np.clip(hp_km, HP_GRID_KM[0], HP_GRID_KM[-1])
    ha = np.clip(np.maximum(ha_km, hp), HA_GRID_KM[0], HA_GRID_KM[-1])
    fi = np.interp(hp, HP_GRID_KM, np.arange(HP_GRID_KM.size))
    fj = np.interp(ha, HA_GRID_KM, np.arange(HA_GRID_KM.size))
    i0 = np.clip(np.floor(fi).astype(int), 0, HP_GRID_KM.size - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, HA_GRID_KM.size - 2)
    di, dj = fi - i0, fj - j0
    vals = (L[i0, j0] * (1 - di) * (1 - dj) + L[i0 + 1, j0] * di * (1 - dj)
            + L[i0, j0 + 1] * (1 - di) * dj + L[i0 + 1, j0 + 1] * di * dj)
    return vals / B


def disposal_perigee_km(ha_km: np.ndarray, B: np.ndarray, target_yr: float) -> np.ndarray:
    """Perigee [km] that gives `target_yr` lifetime while keeping the apogee
    (single perigee-lowering burn at apogee)."""
    hp_trial = np.linspace(HP_GRID_KM[0], HP_GRID_KM[-1], 371)       # 5 km resolution
    out = np.empty(ha_km.size)
    for n, (ha, b) in enumerate(zip(ha_km, B)):
        hp = hp_trial[hp_trial <= ha]
        life = orbit_lifetime_yr(hp, np.full(hp.size, ha), np.full(hp.size, b))
        ok = np.flatnonzero(life <= target_yr)
        out[n] = hp[ok[-1]] if ok.size else HP_GRID_KM[0]
    return out
