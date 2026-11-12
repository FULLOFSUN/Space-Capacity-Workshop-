"""Monte Carlo evolution of the LEO population, 2026 -> end year."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import breakup, collisions
from .atmosphere import solar_cycle_factor
from .constants import (CD, MISSION_LIFETIME_YR, MJD_J2026, RE,
                        REENTRY_MEAN_ALT_KM, REENTRY_PERIGEE_KM)
from .dynamics import decay_step, disposal_perigee_km, orbit_lifetime_yr
from .population import (CATEGORIES, CONSTELLATION, FRAGMENT_CATS,
                         MANOEUVRABLE_CATS, NEW_DEBRIS, Population,
                         load_population)

PC_BIN_EDGES_KM = np.arange(0, 2001, 200)  # 0-200, 200-400, ... 1800-2000
SLOW_DECAY_PERIGEE_KM = 900.0              # above this, decay is updated every 12 steps


@dataclass
class Scenario:
    key: str
    label: str
    intact_csv: str
    pmd_max_years: float
    pmd_success: float = 0.9
    debris_csv: str = "ex1_debris_data.csv"


SCENARIOS = {
    "nfl": Scenario("nfl", "No future launches", "ex1_intact_objs_nfl.csv", 25.0),
    "baseline": Scenario("baseline", "Baseline", "ex1_intact_objs_baseline.csv", 25.0),
    "bonus": Scenario("bonus", "Bonus (5-year PMD)", "ex1_intact_objs_baseline.csv", 5.0),
}


@dataclass
class RunResult:
    counts: pd.DataFrame                    # Year x categories
    collisions: pd.DataFrame                # one row per collision
    pc_bins: dict[int, pd.DataFrame]        # snapshot year -> altitude-bin table
    reentries: pd.Series = field(default_factory=pd.Series)


def _year(mjd: float) -> float:
    return 2026.0 + (mjd - MJD_J2026) / 365.25


def _category_counts(pop: Population, mask: np.ndarray) -> list[int]:
    return [int(np.count_nonzero(mask & (pop.cat == c))) for c in range(len(CATEGORIES))]


def _pc_table(pop: Population, in_orbit: np.ndarray, at_risk: np.ndarray) -> pd.DataFrame:
    """Counts of all objects per 200 km bin, and the summed annual collision
    probability of the objects that can actually collide (`at_risk`, i.e. not
    protected by COLA), each against all other at-risk objects."""
    idx = np.flatnonzero(at_risk)
    occ = collisions.occupancy(pop.a[idx], pop.e[idx])
    rate = collisions.object_rates(occ, np.sqrt(pop.area[idx] / np.pi) / 1e3)
    pc = 1.0 - np.exp(-rate)
    alt_risk = pop.mean_alt_km[idx]

    alt_all = pop.mean_alt_km[in_orbit]
    frag_all = np.isin(pop.cat[in_orbit], FRAGMENT_CATS)
    rows = []
    for lo, hi in zip(PC_BIN_EDGES_KM[:-1], PC_BIN_EDGES_KM[1:]):
        m = (alt_all > lo) & (alt_all <= hi)
        rows.append({"Altitude (>)": f"{hi} km",
                     "Intact objects": int(np.count_nonzero(m & ~frag_all)),
                     "Fragments": int(np.count_nonzero(m & frag_all)),
                     "Cumulative Pc": float(pc[(alt_risk > lo) & (alt_risk <= hi)].sum())})
    return pd.DataFrame(rows)


def _pair_label(cat_i: int, cat_j: int) -> str:
    fi, fj = cat_i in FRAGMENT_CATS, cat_j in FRAGMENT_CATS
    if fi and fj:
        return "Fragment-fragment"
    if fi or fj:
        return "Intact-fragment"
    return "Intact-intact"


def run_once(scn: Scenario, data_dir: Path, end_year: int, snapshot_years: list[int],
             seed: int, steps_per_year: int = 48, solar_amplitude: float = 0.7,
             verbose: bool = False) -> RunResult:
    rng = np.random.default_rng(seed)
    pop = load_population(data_dir / scn.intact_csv, data_dir / scn.debris_csv,
                          rng, scn.pmd_success)

    life_days = MISSION_LIFETIME_YR * 365.25
    manoeuvrable = np.isin(pop.cat, MANOEUVRABLE_CATS)
    # objects whose 5-year life ended before 2026 are already derelict: no PMD
    pop.pmd_done |= ~manoeuvrable | (pop.launch_mjd + life_days < MJD_J2026)

    dt_yr = 1.0 / steps_per_year
    n_steps = int(round((end_year - 2026) * steps_per_year))
    counts, pc_bins, events, reentries = [], {}, [], {}

    for step in range(n_steps + 1):
        mjd = MJD_J2026 + step * dt_yr * 365.25
        year = _year(mjd)
        in_orbit = pop.alive & (pop.launch_mjd <= mjd)
        manoeuvrable = np.isin(pop.cat, MANOEUVRABLE_CATS)
        active = in_orbit & manoeuvrable & (mjd < pop.launch_mjd + life_days)

        # ----- bookkeeping at whole years -------------------------------
        if step % steps_per_year == 0:
            y = 2026 + step // steps_per_year
            counts.append([y] + _category_counts(pop, in_orbit))
            if y in snapshot_years:
                pc_bins[y] = _pc_table(pop, in_orbit, in_orbit & ~active)
            if verbose:
                print(f"  {y}: {int(in_orbit.sum()):6d} objects, {len(events)} collisions so far",
                      flush=True)
        if step == n_steps:
            break

        # ----- end of life: post-mission disposal -------------------------
        # successful PMD lowers the perigee (apogee kept) until the natural
        # lifetime equals the limit; objects already compliant just decay.
        eol = in_orbit & ~active & ~pop.pmd_done
        if eol.any():
            idx = np.flatnonzero(eol & pop.pmd_ok)
            if idx.size:
                B = CD * pop.area[idx] / pop.mass[idx]
                hp, ha = pop.perigee_km[idx], pop.apogee_km[idx]
                needs = orbit_lifetime_yr(hp, ha, B) > scn.pmd_max_years
                if needs.any():
                    n = idx[needs]
                    hp_new = np.minimum(disposal_perigee_km(ha[needs], B[needs],
                                                            scn.pmd_max_years), hp[needs])
                    r_p, r_a = RE + hp_new * 1e3, RE + ha[needs] * 1e3
                    pop.a[n] = 0.5 * (r_p + r_a)
                    pop.e[n] = (r_a - r_p) / (r_a + r_p)
            pop.pmd_done |= eol

        # ----- drag decay of everything not station-keeping --------------
        rho_f = solar_cycle_factor(year, solar_amplitude)
        passive = in_orbit & ~active
        low = passive & (pop.perigee_km < SLOW_DECAY_PERIGEE_KM)
        high = passive & ~low
        for mask, mult in ((low, 1), (high, 12 if step % 12 == 0 else 0)):
            if mult:
                p = np.flatnonzero(mask)
                pop.a[p], pop.e[p] = decay_step(pop.a[p], pop.e[p], pop.area[p],
                                                pop.mass[p], dt_yr * mult, rho_f)
        gone = in_orbit & ((pop.perigee_km < REENTRY_PERIGEE_KM)
                           | (pop.mean_alt_km < REENTRY_MEAN_ALT_KM))
        if gone.any():
            pop.alive[gone] = False
            reentries[int(year)] = reentries.get(int(year), 0) + int(gone.sum())

        # ----- collisions (active satellites avoid everything: perfect COLA)
        elig = np.flatnonzero(pop.alive & (pop.launch_mjd <= mjd) & ~active)
        occ = collisions.occupancy(pop.a[elig], pop.e[elig])
        radius_km = np.sqrt(pop.area[elig] / np.pi) / 1e3
        n_coll = rng.poisson(collisions.shell_rates(occ, radius_km) * dt_yr)

        for k in np.flatnonzero(n_coll):
            for _ in range(n_coll[k]):
                ii, jj = collisions.sample_pair(occ, k, radius_km, rng)
                i, j = elig[ii], elig[jj]
                if not (pop.alive[i] and pop.alive[j]):
                    continue                       # already destroyed this step
                t, pr = (i, j) if pop.mass[i] >= pop.mass[j] else (j, i)
                cat = breakup.is_catastrophic(pop.mass[t], pop.mass[pr])
                lo, hi = collisions.SHELL_EDGES_KM[k], collisions.SHELL_EDGES_KM[k + 1]
                alt_km = rng.uniform(lo, hi)
                frags = breakup.fragment_cloud(pop.mass[t], pop.mass[pr], pop.area[t],
                                               cat, RE + alt_km * 1e3, pop.inc[t], rng)
                nf = frags["a"].size
                consts = sorted({pop.group[x] for x in (t, pr) if pop.cat[x] == CONSTELLATION})
                events.append({
                    "Epoch": round(year, 3),
                    "Altitude [km]": round(alt_km, 1),
                    "Type of collision": "Catastrophic" if cat else "Non Catastrophic",
                    "Objects involved": _pair_label(pop.cat[t], pop.cat[pr]),
                    "Constallations involved": ", ".join(consts) if consts else "None",
                    "Fragments generated": int(nf),
                    "Categories involved": f"{CATEGORIES[pop.cat[t]]} vs {CATEGORIES[pop.cat[pr]]}",
                })
                if cat:
                    pop.alive[[t, pr]] = False
                else:                              # projectile destroyed, target cratered
                    pop.alive[pr] = False
                    pop.mass[t] = max(pop.mass[t] - pop.mass[pr], 0.05)
                pop.append(a=frags["a"], e=frags["e"], inc=frags["inc"],
                           area=frags["area"], mass=frags["mass"],
                           launch_mjd=np.full(nf, mjd), cat=np.full(nf, NEW_DEBRIS),
                           group=np.full(nf, "Fragment", dtype=object),
                           pmd_ok=np.zeros(nf, bool), pmd_done=np.ones(nf, bool),
                           alive=np.ones(nf, bool))

    counts_df = pd.DataFrame(counts, columns=["Year"] + CATEGORIES)
    counts_df["Total"] = counts_df[CATEGORIES].sum(axis=1)
    coll_df = pd.DataFrame(events, columns=["Epoch", "Altitude [km]", "Type of collision",
                                            "Objects involved", "Constallations involved",
                                            "Fragments generated", "Categories involved"])
    coll_df.insert(0, "Id", np.arange(1, len(coll_df) + 1))
    return RunResult(counts_df, coll_df, pc_bins, pd.Series(reentries, name="reentries"))
