"""Summary figures for one or more scenarios."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .population import CATEGORIES

CAT_COLORS = {
    "Old Debris": "#7f7f7f", "New Debris generated": "#d62728", "Payloads": "#1f77b4",
    "Rocket Bodies": "#ff7f0e", "MRO": "#9467bd", "Constellations": "#2ca02c",
}
SCN_COLORS = {"nfl": "#4c72b0", "baseline": "#dd8452", "bonus": "#55a868"}


def plot_scenario(key: str, label: str, counts: list[pd.DataFrame],
                  colls: list[pd.DataFrame], out_dir: Path) -> Path:
    years = counts[0]["Year"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    stack = np.mean([c[CATEGORIES].to_numpy() for c in counts], axis=0)
    ax1.stackplot(years, stack.T, labels=CATEGORIES,
                  colors=[CAT_COLORS[c] for c in CATEGORIES], alpha=0.9)
    ax1.set(title=f"{label}: objects in orbit (mean of {len(counts)} MC runs)",
            xlabel="Year", ylabel="Number of objects")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    ax1.grid(alpha=0.3)

    for c in colls:
        cum = [int((c["Epoch"] < y).sum()) for y in years]
        ax2.step(years, cum, where="post", color="0.6", lw=0.8, alpha=0.7)
    mean_cum = np.mean([[int((c["Epoch"] < y).sum()) for y in years] for c in colls], axis=0)
    ax2.step(years, mean_cum, where="post", color="k", lw=2, label="mean")
    ax2.set(title="Cumulative collisions (grey: individual MC runs)",
            xlabel="Year", ylabel="Collisions")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"{key}_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_comparison(results: dict[str, dict], out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for key, res in results.items():
        years = res["counts"][0]["Year"].to_numpy()
        tot = np.array([c["Total"].to_numpy() for c in res["counts"]])
        col = SCN_COLORS.get(key, None)
        ax1.plot(years, tot.mean(0), color=col, lw=2, label=res["label"])
        ax1.fill_between(years, tot.min(0), tot.max(0), color=col, alpha=0.2)
        cum = np.array([[int((c["Epoch"] < y).sum()) for y in years] for c in res["collisions"]])
        ax2.plot(years, cum.mean(0), color=col, lw=2, label=res["label"])
        ax2.fill_between(years, cum.min(0), cum.max(0), color=col, alpha=0.2)
    ax1.set(title="Total objects in orbit (band: MC min-max)", xlabel="Year", ylabel="Objects")
    ax1.set_ylim(bottom=0)
    ax2.set(title="Cumulative collisions (band: MC min-max)", xlabel="Year", ylabel="Collisions")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "scenario_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_outputs(folder: Path) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Read counts and collisions back from a scenario's output folder."""
    n = pd.read_excel(folder / "output_N_objs.xlsx", sheet_name=None)
    c = pd.read_excel(folder / "output_collisions.xlsx", sheet_name=None)
    counts = [df.assign(Total=df[CATEGORIES].sum(axis=1)) for k, df in n.items() if k.startswith("MC")]
    colls = [df for k, df in c.items() if k.startswith("MC")]
    return counts, colls


if __name__ == "__main__":
    # Re-draw figures from saved outputs:  python -m debris2050.plots outputs/nfl [outputs/baseline ...]
    import sys
    results = {}
    for arg in sys.argv[1:]:
        folder = Path(arg)
        counts, colls = load_outputs(folder)
        print("wrote", plot_scenario(folder.name, folder.name, counts, colls, folder))
        results[folder.name] = {"label": folder.name, "counts": counts, "collisions": colls}
    if len(results) > 1:
        print("wrote", plot_comparison(results, Path(sys.argv[1]).parent))
