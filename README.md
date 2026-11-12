# leo-debris-2050

**A short-horizon (2026–2050) Monte Carlo model of the LEO debris environment, built for the ESA/MIT
long-term simulation exercise.**

The workshop exercise asks for 100-year runs (2026–2126). This repository runs the same three scenarios over
the next 25 years, which is the horizon that matters for current operators and policy. It writes the results
in the workshop's own output format (`output_N_objs.xlsx`, `output_collisions.xlsx`, `output_pc_bins.xlsx`).

![No-future-launches scenario, 10 Monte Carlo runs](docs/nfl_summary.png)

## Scenarios

| Key | Scenario | Population file | Post-mission disposal (PMD) |
|---|---|---|---|
| `nfl` | No future launches | `ex1_intact_objs_nfl.csv` | 90% success, re-entry within 25 years |
| `baseline` | Baseline: SARIMAX launch trend + Starlink and Guowang | `ex1_intact_objs_baseline.csv` | 90% success, re-entry within 25 years |
| `bonus` | Bonus: stricter disposal | `ex1_intact_objs_baseline.csv` | 90% success, re-entry within **5** years |

Common rules, as in the exercise: 5-year mission lifetime; perfect collision avoidance (COLA) while
operational and none afterwards; no explosions; no active debris removal; no objects added beyond those in the
files; fragments 10 cm and larger.

## Quick start

```bash
git clone https://github.com/<you>/leo-debris-2050.git
cd leo-debris-2050
pip install -r requirements.txt
# put the three input CSVs in data/  (see data/README.md)

python run.py --scenario nfl --runs 2 --end-year 2030          # 1-minute smoke test
python run.py --scenario nfl                                   # 10 MC runs, 2026-2050
python run.py --scenario nfl baseline bonus --runs 10 --workers 4
python -m debris2050.plots outputs/nfl outputs/baseline outputs/bonus   # redraw figures
pytest -q
```

Outputs go to `outputs/<scenario>/`:

| File | Sheets | Content |
|---|---|---|
| `output_N_objs.xlsx` | `Mean`, `MC 1`…`MC n` | Objects in orbit on 1 January of each year: Old Debris, New Debris generated, Payloads, Rocket Bodies, MRO, Constellations, Total |
| `output_collisions.xlsx` | `MC 1`…`MC n`, `IDs` | One row per collision: epoch, altitude, catastrophic or not, intact/fragment pairing, constellations involved, fragments generated, plus the category pair (e.g. "Payloads vs Old Debris") |
| `output_pc_bins.xlsx` | `Mean`, `MC 1`…`MC n` | Per 200 km bin at snapshot years (default 2026, 2030, 2035, 2040, 2045, 2050): intact objects, fragments, cumulative collision probability |
| `<scenario>_summary.png` | | Population by category and cumulative collisions |

Useful options: `--end-year 2051` matches the template's first 25-year snapshot; `--snapshots 2026 2040 2050`;
`--solar-amplitude 0` switches off the solar cycle; `--steps-per-year 24` halves the run time.

## How the model works

The population is a set of objects, each with semi-major axis, eccentricity, area, mass, launch date and
category. Every time step (default 1/48 year), the model does four things:

1. **Launch and operations.** An object enters the simulation on its launch MJD. Payloads and constellation
   satellites are *active* for 5 years: they hold their orbit and avoid every collision.
2. **Post-mission disposal.** At end of life, 90% of satellites perform PMD. If the natural lifetime exceeds
   the limit (25 or 5 years), a single burn at apogee lowers the perigee until the lifetime equals the limit.
   The other 10% are left where they are.
3. **Drag decay.** Every non-active object decays with the King-Hele orbit-averaged equations,

   da/dt = −B ρ<sub>p</sub> √(μa) e<sup>−c</sup>[I₀(c) + 2e I₁(c)],  de/dt = −B ρ<sub>p</sub> √(μ/a) e<sup>−c</sup>[I₁(c) + ½e(I₀(c) + I₂(c))]

   with B = C<sub>D</sub>A/m, C<sub>D</sub> = 2.2, c = ae/H, and ρ<sub>p</sub> the density at perigee.
   Density follows Vallado's exponential atmosphere, modulated by an 11-year solar cycle (×2 at maximum,
   ×0.5 at minimum, peak 2025). Objects are removed once their perigee falls below 120 km.
4. **Collisions.** LEO is divided into 50 km shells. Each object spends a fraction f<sub>ik</sub> of its orbit in
   shell k (from Kepler's equation). The expected number of collisions per year in a shell is

   λ<sub>k</sub> = (v<sub>rel</sub> / V<sub>k</sub>) Σ<sub>i&lt;j</sub> f<sub>ik</sub> f<sub>jk</sub> π (r<sub>i</sub> + r<sub>j</sub>)²,  v<sub>rel</sub> = 10 km/s

   This pair sum is evaluated in O(N) with weighted moments. The number of collisions per step is drawn from
   a Poisson distribution, and each colliding pair is sampled with exactly those pair weights.
   Each collision goes through the **NASA Standard Breakup Model**: catastrophic if the energy-to-mass ratio
   is at least 40 J/g, N(L<sub>c</sub>) = 0.1 M<sup>0.75</sup> L<sub>c</sub><sup>−1.71</sup> fragments ≥ 10 cm,
   area-to-mass from the bimodal distribution, and a Δv kick that sets each fragment's new orbit. Total
   fragment mass never exceeds the colliding mass.

**Pc definition.** "Cumulative Pc" in a bin is the sum, over the objects in that bin, of each object's annual
probability of colliding with any other object. Active (COLA-protected) satellites are excluded, so Pc is
consistent with the collisions actually simulated. Roughly, half the summed Pc is the expected number of
collisions per year in that bin.

### Checks

`pytest` covers the building blocks. The table lifetimes match the exact circular-orbit integral to 0.1%.
The O(N) collision rates equal a brute-force pair sum to machine precision. Disposal orbits meet their target
lifetime. Fragment clouds respect the mass budget.

## Results: no future launches, 2026–2050 (10 MC runs)

| Year | Old debris | New debris | Payloads | Rocket bodies | MRO | Constellations | **Total** |
|---|---|---|---|---|---|---|---|
| 2026 | 14,059 | 0 | 3,927 | 1,002 | 383 | 10,572 | **29,943** |
| 2030 | 13,692 | 897 | 3,740 | 952 | 363 | 6,286 | **25,930** |
| 2035 | 13,495 | 1,999 | 2,966 | 919 | 342 | 1,197 | **20,917** |
| 2040 | 13,235 | 3,512 | 2,489 | 876 | 310 | 1,158 | **21,580** |
| 2050 | 12,891 | 5,643 | 2,070 | 823 | 268 | 658 | **22,352** |

- **Collisions:** 12.4 per run on average (range 4–19), about 0.5 per year. 85% are catastrophic, and most occur
  between 600 and 1,000 km.
- **The environment is not self-cleaning.** Low shells empty as the Starlink fleet re-enters within about five
  years, yet the total stops falling by the mid-2030s. Collision fragments then accumulate at 600–1,000 km,
  where drag is too weak to remove them within the horizon.
- **The 2030 risk spike** (summed Pc ≈ 3 vs 0.7 in 2026) comes from the end-of-life wave of constellation
  satellites. The rules give them no collision avoidance once they stop operating, and those already within
  the 25-year limit are not manoeuvred down.

The full Excel outputs for this run are in `example_outputs/nfl/`. Run `baseline` and `bonus` with the full launch file to see how much the 5-year rule changes this.

## Limitations

This is a teaching- and comparison-grade model, not a replacement for MASTER, DELTA, LEGEND or MOCAT:

- Shell-averaged (kinetic-gas) collision rates with one mean impact velocity. There is no orbital-plane
  geometry, and RAAN, AoP and M are ignored.
- A static exponential atmosphere with a sinusoidal solar cycle, instead of NRLMSISE-00 driven by F10.7 and Ap
  forecasts.
- Fragments inherit the target's inclination. Δv changes only a and e.
- Non-catastrophic collisions destroy the projectile and reduce the target's mass, without changing its orbit.

## References

- Johnson, N. L. et al. (2001). NASA's new breakup model of EVOLVE 4.0. *Adv. Space Res.* 28(9), 1377–1384.
- King-Hele, D. (1987). *Satellite Orbits in an Atmosphere: Theory and Applications*. Blackie.
- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*, 4th ed. (Table 8-4).
- Liou, J.-C. et al. (2003). Improved long-term collision probability calculation in LEO (CUBE). *Adv. Space Res.* 31(11).
- ESA Space Debris Office (2025). *ESA's Annual Space Environment Report*.

## License

MIT
