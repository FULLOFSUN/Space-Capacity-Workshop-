"""NASA Standard Breakup Model (Johnson et al. 2001) for collisions,
restricted to fragments >= 10 cm to match the input debris population.
"""
from __future__ import annotations

import numpy as np

from .constants import (CATASTROPHIC_EMR_J_PER_G, LC_MIN_M, MU, RE,
                        V_REL_KM_S)

POWER_LAW = 1.71


def is_catastrophic(m_target: float, m_projectile: float,
                    v_km_s: float = V_REL_KM_S) -> bool:
    """Energy-to-mass ratio test: EMR = 0.5 m_p v^2 / m_t >= 40 J/g."""
    emr = 0.5 * m_projectile * (v_km_s * 1e3) ** 2 / (m_target * 1e3)
    return emr >= CATASTROPHIC_EMR_J_PER_G


def n_fragments(m_target: float, m_projectile: float, catastrophic: bool,
                v_km_s: float = V_REL_KM_S, lc: float = LC_MIN_M) -> float:
    """N(>Lc) = 0.1 M^0.75 Lc^-1.71."""
    M = (m_target + m_projectile) if catastrophic else m_projectile * v_km_s ** 2
    return 0.1 * M ** 0.75 * lc ** (-POWER_LAW)


def _log_am_spacecraft(lc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """log10(A/M) for spacecraft fragments with Lc > 11 cm (bimodal normal)."""
    x = np.log10(lc)
    alpha = np.where(x <= -1.95, 0.0, np.where(x < 0.55, 0.3 + 0.4 * (x + 1.2), 1.0))
    mu1 = np.where(x <= -1.1, -0.6, np.where(x < 0.0, -0.6 - 0.318 * (x + 1.1), -0.95))
    s1 = np.where(x <= -1.3, 0.1, np.where(x < -0.3, 0.1 + 0.2 * (x + 1.3), 0.3))
    mu2 = np.where(x <= -0.7, -1.2, np.where(x < -0.1, -1.2 - 1.333 * (x + 0.7), -2.0))
    s2 = np.where(x <= -0.5, 0.5, np.where(x < -0.3, 0.5 - (x + 0.5), 0.3))
    first = rng.random(lc.size) < np.clip(alpha, 0, 1)
    return np.where(first, rng.normal(mu1, s1), rng.normal(mu2, s2))


def fragment_cloud(m_target: float, m_projectile: float, area_target: float,
                   catastrophic: bool, r_m: float, inc_deg: float,
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate fragments >= Lc and their orbits after a collision at radius r."""
    n = rng.poisson(n_fragments(m_target, m_projectile, catastrophic))
    if n == 0:
        return {k: np.empty(0) for k in ("a", "e", "inc", "area", "mass")}

    # sizes from the truncated power law between Lc and the parent's size
    l_max = max(2 * LC_MIN_M, np.sqrt(4 * area_target / np.pi))
    u = rng.random(n)
    k = -POWER_LAW
    lc = (LC_MIN_M ** k + u * (l_max ** k - LC_MIN_M ** k)) ** (1 / k)

    area = np.where(lc > 0.00167, 0.556945 * lc ** 2.0047077, 0.540424 * lc ** 2)
    am = 10 ** _log_am_spacecraft(lc, rng)
    mass = area / am

    # mass budget: never create more fragment mass than is available
    budget = (m_target + m_projectile) if catastrophic else m_projectile * V_REL_KM_S ** 2
    budget = min(budget, m_target + m_projectile)
    order = rng.permutation(n)
    keep = order[np.cumsum(mass[order]) <= budget]
    lc, area, mass, am = lc[keep], area[keep], mass[keep], am[keep]
    n = lc.size

    # delta-v: log10(dv [m/s]) ~ N(0.9 chi + 2.9, 0.4), chi = log10(A/M)
    dv = 10 ** rng.normal(0.9 * np.log10(am) + 2.9, 0.4)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    dvx, dvy, dvz = (direction * dv[:, None]).T  # radial, along-track, cross-track

    v_c = np.sqrt(MU / r_m)
    vt, vr, vn = v_c + dvy, dvx, dvz
    v2 = vt ** 2 + vr ** 2 + vn ** 2
    a = 1.0 / (2.0 / r_m - v2 / MU)
    h = r_m * np.sqrt(vt ** 2 + vn ** 2)
    with np.errstate(invalid="ignore"):
        e = np.sqrt(np.clip(1.0 - h ** 2 / (MU * a), 0.0, None))
    bound = (a > 0) & (e < 1.0) & (a * (1 - e) > RE)

    return {"a": a[bound], "e": e[bound], "inc": np.full(bound.sum(), inc_deg),
            "area": area[bound], "mass": mass[bound]}
