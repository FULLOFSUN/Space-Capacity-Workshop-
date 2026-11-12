"""Shell-based (kinetic-gas) collision rates and pair sampling.

Space between 0 and 2000 km is split into spherical shells. Each object
spends a fraction f_ik of its orbit in shell k (from Kepler's equation), so
its spatial density there is f_ik / V_k. The expected number of collisions
per unit time in shell k is

    lambda_k = (v_rel / V_k) * sum_{i<j} f_ik f_jk * pi (r_i + r_j)^2

with r_i = sqrt(A_i / pi). The double sum is evaluated in O(N) through
weighted moments, and colliding pairs are sampled exactly from the same
weights. Fractions are stored sparsely as (object, shell, fraction) triplets
because most LEO orbits cross only one to three shells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import LEO_TOP_KM, RE, SEC_PER_YEAR, SHELL_WIDTH_KM, V_REL_KM_S

SHELL_EDGES_KM = np.arange(0.0, LEO_TOP_KM + SHELL_WIDTH_KM, SHELL_WIDTH_KM)
N_SHELLS = SHELL_EDGES_KM.size - 1
_R_EDGES_KM = RE / 1e3 + SHELL_EDGES_KM
SHELL_VOLUME_KM3 = 4.0 / 3.0 * np.pi * (_R_EDGES_KM[1:] ** 3 - _R_EDGES_KM[:-1] ** 3)
V_REL_KM_YR = V_REL_KM_S * SEC_PER_YEAR
_RATE_COEF = np.pi * V_REL_KM_YR / SHELL_VOLUME_KM3


def time_below(r_km, a_km, e):
    """Fraction of the orbital period spent at radius < r (broadcasts)."""
    r_km, a_km, e = np.broadcast_arrays(np.asarray(r_km, float),
                                        np.asarray(a_km, float), np.asarray(e, float))
    rp, ra = a_km * (1 - e), a_km * (1 + e)
    safe_e = np.where(e > 1e-12, e, 1.0)
    E = np.arccos(np.clip((1.0 - r_km / a_km) / safe_e, -1.0, 1.0))
    frac = (E - e * np.sin(E)) / np.pi
    return np.where(r_km <= rp, 0.0, np.where(r_km >= ra, 1.0, frac))


@dataclass
class Occupancy:
    """Sparse shell occupancy: object `obj[n]` spends `w[n]` of its time in `shell[n]`."""
    obj: np.ndarray
    shell: np.ndarray
    w: np.ndarray
    n_objects: int


def occupancy(a_m: np.ndarray, e: np.ndarray) -> Occupancy:
    a_km = a_m / 1e3
    rp, ra = a_km * (1 - e), a_km * (1 + e)
    kp = np.searchsorted(_R_EDGES_KM, rp, side="right") - 1      # shell holding perigee
    ka = np.searchsorted(_R_EDGES_KM, ra, side="right") - 1      # shell holding apogee
    inside = (kp < N_SHELLS) & (ka >= 0)                         # orbit touches LEO shells
    idx = np.flatnonzero(inside)
    kp_c = np.clip(kp[idx], 0, N_SHELLS - 1)
    ka_c = np.clip(ka[idx], 0, N_SHELLS - 1)

    n_sh = ka_c - kp_c + 1                                       # shells crossed
    obj = np.repeat(idx, n_sh)
    first = np.repeat(np.cumsum(n_sh) - n_sh, n_sh)
    shell = np.repeat(kp_c, n_sh) + (np.arange(obj.size) - first)

    lo = time_below(_R_EDGES_KM[shell], a_km[obj], e[obj])
    hi = time_below(_R_EDGES_KM[shell + 1], a_km[obj], e[obj])
    w = hi - lo
    keep = w > 0
    return Occupancy(obj[keep], shell[keep], w[keep], a_m.size)


def _moments(occ: Occupancy, radius_km: np.ndarray):
    r = radius_km[occ.obj]
    w = occ.w
    S0 = np.bincount(occ.shell, w, N_SHELLS)
    S1 = np.bincount(occ.shell, w * r, N_SHELLS)
    S2 = np.bincount(occ.shell, w * r * r, N_SHELLS)
    Q = np.bincount(occ.shell, w * w * r * r, N_SHELLS)
    return S0, S1, S2, Q


def shell_rates(occ: Occupancy, radius_km: np.ndarray) -> np.ndarray:
    """Expected collisions per year in each shell (lambda_k)."""
    S0, S1, S2, Q = _moments(occ, radius_km)
    return _RATE_COEF * np.maximum(S0 * S2 + S1 * S1 - 2.0 * Q, 0.0)


def object_rates(occ: Occupancy, radius_km: np.ndarray) -> np.ndarray:
    """Annual collision rate of each object against all others in the set."""
    S0, S1, S2, _ = _moments(occ, radius_km)
    r, w, k = radius_km[occ.obj], occ.w, occ.shell
    others = r * r * S0[k] + 2 * r * S1[k] + S2[k] - w * (2 * r) ** 2
    return np.bincount(occ.obj, _RATE_COEF[k] * w * others, occ.n_objects)


def sample_pair(occ: Occupancy, k: int, radius_km: np.ndarray,
                rng: np.random.Generator) -> tuple[int, int]:
    """Draw (i, j) in shell k with probability proportional to f_i f_j (r_i + r_j)^2."""
    m = occ.shell == k
    obj, w = occ.obj[m], occ.w[m]
    r = radius_km[obj]
    S0, S1, S2 = w.sum(), (w * r).sum(), (w * r * r).sum()
    p_cross = S1 * S1 / (S0 * S2 + S1 * S1)
    for _ in range(100):
        if rng.random() < p_cross:            # term 2 r_i r_j
            p = w * r
            i, j = rng.choice(obj.size, size=2, p=p / p.sum())
        else:                                 # terms r_i^2 + r_j^2
            p = w * r * r
            i = rng.choice(obj.size, p=p / p.sum())
            j = rng.choice(obj.size, p=w / S0)
        if i != j:
            return int(obj[i]), int(obj[j])
    raise RuntimeError("could not sample a distinct collision pair")
