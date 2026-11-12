"""Static exponential atmosphere, a simple solar-cycle modulation, and
circular-orbit lifetime look-ups used for post-mission disposal (PMD).

The density table is the piecewise exponential model from Vallado,
*Fundamentals of Astrodynamics and Applications* (Table 8-4), valid for
mean solar conditions.
"""
from __future__ import annotations

import numpy as np

from .constants import MU, RE, SEC_PER_YEAR, REENTRY_MEAN_ALT_KM

# base altitude [km], nominal density [kg/m^3], scale height [km]
_VALLADO = np.array([
    [0, 1.225, 7.249], [25, 3.899e-2, 6.349], [30, 1.774e-2, 6.682],
    [40, 3.972e-3, 7.554], [50, 1.057e-3, 8.382], [60, 3.206e-4, 7.714],
    [70, 8.770e-5, 6.549], [80, 1.905e-5, 5.799], [90, 3.396e-6, 5.382],
    [100, 5.297e-7, 5.877], [110, 9.661e-8, 7.263], [120, 2.438e-8, 9.473],
    [130, 8.484e-9, 12.636], [140, 3.845e-9, 16.149], [150, 2.070e-9, 22.523],
    [180, 5.464e-10, 29.740], [200, 2.789e-10, 37.105], [250, 7.248e-11, 45.546],
    [300, 2.418e-11, 53.628], [350, 9.518e-12, 53.298], [400, 3.725e-12, 58.515],
    [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822], [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64], [900, 5.245e-15, 181.05],
    [1000, 3.019e-15, 268.00],
])
_H0, _RHO0, _SH = _VALLADO[:, 0], _VALLADO[:, 1], _VALLADO[:, 2]


def density(h_km: np.ndarray) -> np.ndarray:
    """Mean-solar-activity density [kg/m^3] at altitude h [km]."""
    h = np.clip(np.asarray(h_km, dtype=float), 0.0, None)
    idx = np.clip(np.searchsorted(_H0, h, side="right") - 1, 0, len(_H0) - 1)
    return _RHO0[idx] * np.exp(-(h - _H0[idx]) / _SH[idx])


def scale_height_km(h_km: np.ndarray) -> np.ndarray:
    h = np.clip(np.asarray(h_km, dtype=float), 0.0, None)
    idx = np.clip(np.searchsorted(_H0, h, side="right") - 1, 0, len(_H0) - 1)
    return _SH[idx]


def solar_cycle_factor(year: float, amplitude: float = 0.7,
                       max_year: float = 2025.0, period: float = 11.0) -> float:
    """Multiplicative density factor exp(A cos(2 pi (t - t_max) / P)).

    With the default A = 0.7 the density is ~2x the mean at solar maximum
    and ~0.5x at solar minimum. Cycle 25 peaked around 2024-2025.
    Set amplitude = 0 to switch the cycle off.
    """
    return float(np.exp(amplitude * np.cos(2.0 * np.pi * (year - max_year) / period)))


# ---------------------------------------------------------------------------
# Circular-orbit lifetime: t(h) = G(h) / B with B = Cd*A/m and
#   G(h) = integral_{h_reentry}^{h} dh / (rho(h) sqrt(mu (Re + h)))
# Precomputed once on a fine grid so PMD targets are a fast interpolation.
# ---------------------------------------------------------------------------
_GRID_KM = np.linspace(REENTRY_MEAN_ALT_KM, 2500.0, 20000)
_INTEGRAND = 1.0 / (density(_GRID_KM) * np.sqrt(MU * (RE + _GRID_KM * 1e3)))  # [s m^2/kg per m]
_G = np.concatenate([[0.0], np.cumsum(0.5 * (_INTEGRAND[1:] + _INTEGRAND[:-1])
                                      * np.diff(_GRID_KM) * 1e3)])  # [s kg/m^2]


def circular_lifetime_yr(h_km: np.ndarray, bstar_m2_per_kg: np.ndarray) -> np.ndarray:
    """Approximate natural lifetime [yr] of a circular orbit at h with B = Cd A/m."""
    g = np.interp(np.asarray(h_km, dtype=float), _GRID_KM, _G)
    return g / np.asarray(bstar_m2_per_kg) / SEC_PER_YEAR


def altitude_for_lifetime_km(lifetime_yr: float, bstar_m2_per_kg: np.ndarray) -> np.ndarray:
    """Circular altitude [km] whose natural lifetime equals `lifetime_yr`."""
    target = lifetime_yr * SEC_PER_YEAR * np.asarray(bstar_m2_per_kg)
    return np.interp(target, _G, _GRID_KM)
