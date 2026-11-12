#!/usr/bin/env python3
"""Run the short-horizon (2026-2050) LEO debris scenarios.

Examples
--------
    python run.py --scenario nfl                       # 10 MC runs, 2026-2050
    python run.py --scenario nfl baseline bonus --runs 10 --workers 4
    python run.py --scenario nfl --runs 2 --end-year 2035   # quick test
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from debris2050 import outputs, plots
from debris2050.simulation import SCENARIOS, run_once

NOTE = (
    "Instructions / model notes:\n"
    "- Short-horizon run ({start}-{end}) of the ESA/MIT long-term exercise, scenario: {label}.\n"
    "- Model: debris2050 (shell-based Monte Carlo, King-Hele drag decay, NASA SBM breakups).\n"
    "- {runs} Monte Carlo runs; 'Mean' is the average over runs.\n"
    "- Categories: Payloads = 'Payload'; Constellations = every named constellation; "
    "Old Debris = MASTER input fragments; New Debris = fragments from simulated collisions.\n"
    "- Cumulative Pc = sum over objects in the bin of each object's annual collision probability "
    "against all other objects that are not protected by COLA (active satellites excluded).\n"
    "- Altitude bins use mean altitude (a - Re), from the previous limit to the stated value."
)


def _one(args):
    key, data_dir, end_year, snaps, seed, spy, solar = args
    return run_once(SCENARIOS[key], Path(data_dir), end_year, snaps, seed,
                    steps_per_year=spy, solar_amplitude=solar)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", nargs="+", default=["nfl"], choices=list(SCENARIOS))
    ap.add_argument("--runs", type=int, default=10, help="Monte Carlo runs per scenario (>=10 for the workshop)")
    ap.add_argument("--end-year", type=int, default=2050)
    ap.add_argument("--snapshots", type=int, nargs="+", default=None,
                    help="years for the Pc tables (default: 2026, 2030, ... end year)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--steps-per-year", type=int, default=48)
    ap.add_argument("--solar-amplitude", type=float, default=0.7,
                    help="solar-cycle density modulation; 0 switches it off")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    a = ap.parse_args()

    snaps = a.snapshots or sorted({2026, *range(2030, a.end_year + 1, 5), a.end_year})
    data_dir, out_root = Path(a.data_dir), Path(a.out_dir)
    comparison = {}

    for key in a.scenario:
        scn = SCENARIOS[key]
        for f in (scn.intact_csv, scn.debris_csv):
            if not (data_dir / f).exists():
                raise SystemExit(f"Missing input file: {data_dir / f}  (see data/README.md)")
        out = out_root / key
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {scn.label}: {a.runs} runs, 2026-{a.end_year} ===", flush=True)
        t0 = time.time()
        jobs = [(key, str(data_dir), a.end_year, snaps, a.seed + i, a.steps_per_year,
                 a.solar_amplitude) for i in range(a.runs)]
        if a.workers > 1:
            with ProcessPoolExecutor(a.workers) as ex:
                results = list(ex.map(_one, jobs))
        else:
            results = []
            for i, j in enumerate(jobs):
                results.append(_one(j))
                print(f"  run {i + 1}/{a.runs}: {len(results[-1].collisions)} collisions, "
                      f"{int(results[-1].counts['Total'].iloc[-1])} objects in {a.end_year}", flush=True)

        note = NOTE.format(start=2026, end=a.end_year, label=scn.label, runs=a.runs)
        counts = [r.counts for r in results]
        colls = [r.collisions for r in results]
        outputs.write_n_objs(counts, out / "output_N_objs.xlsx", note)
        outputs.write_collisions(colls, out / "output_collisions.xlsx", note)
        outputs.write_pc_bins([r.pc_bins for r in results], out / "output_pc_bins.xlsx", note)
        plots.plot_scenario(key, scn.label, counts, colls, out)

        mean_total = pd.concat(counts).groupby("Year")["Total"].mean()
        n_coll = [len(c) for c in colls]
        print(f"  done in {time.time() - t0:.0f} s | objects {mean_total.iloc[0]:.0f} -> "
              f"{mean_total.iloc[-1]:.0f} | collisions per run: mean {sum(n_coll) / len(n_coll):.1f} "
              f"(min {min(n_coll)}, max {max(n_coll)}) | outputs in {out}/")
        comparison[key] = {"label": scn.label, "counts": counts, "collisions": colls}

    if len(comparison) > 1:
        print("comparison figure:", plots.plot_comparison(comparison, out_root))


if __name__ == "__main__":
    main()
