"""Loading the exercise CSV files into a structure-of-arrays population."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import RE, REENTRY_PERIGEE_KM

# Output categories, in the column order of output_N_objs.xlsx
CATEGORIES = ["Old Debris", "New Debris generated", "Payloads",
              "Rocket Bodies", "MRO", "Constellations"]
OLD_DEBRIS, NEW_DEBRIS, PAYLOAD, ROCKET_BODY, MRO, CONSTELLATION = range(6)
FRAGMENT_CATS = (OLD_DEBRIS, NEW_DEBRIS)
MANOEUVRABLE_CATS = (PAYLOAD, CONSTELLATION)


def classify(object_type: str) -> int:
    """Map the 'object type' column to an output category.

    'Payload', 'Rocket Body' and 'MRO' map directly; every other label in the
    files (Starlink, OneWeb, Kuiper, Guowang, ...) is a constellation.
    """
    t = str(object_type).strip().lower()
    if t == "payload":
        return PAYLOAD
    if t in ("rocket body", "rocket_body", "rb"):
        return ROCKET_BODY
    if t in ("mro", "mission related object"):
        return MRO
    if t in ("debris", "fragment"):
        return OLD_DEBRIS
    return CONSTELLATION


@dataclass
class Population:
    a: np.ndarray            # semi-major axis [m]
    e: np.ndarray            # eccentricity [-]
    inc: np.ndarray          # inclination [deg]
    area: np.ndarray         # average cross-section [m^2]
    mass: np.ndarray         # [kg]
    launch_mjd: np.ndarray   # launch epoch (MJD); old debris -> -inf
    cat: np.ndarray          # output category (int)
    group: np.ndarray        # constellation / type label (str)
    pmd_ok: np.ndarray       # bool: will PMD succeed at end of life?
    pmd_done: np.ndarray     # bool: has the EOL logic already been applied?
    alive: np.ndarray        # bool: still in orbit
    names: list = field(default_factory=list)

    def __len__(self) -> int:
        return self.a.size

    @property
    def perigee_km(self) -> np.ndarray:
        return (self.a * (1.0 - self.e) - RE) / 1e3

    @property
    def apogee_km(self) -> np.ndarray:
        return (self.a * (1.0 + self.e) - RE) / 1e3

    @property
    def mean_alt_km(self) -> np.ndarray:
        return (self.a - RE) / 1e3

    def append(self, **cols) -> None:
        for k, v in cols.items():
            setattr(self, k, np.concatenate([getattr(self, k), np.asarray(v)]))


def _read_intact(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["object type", "sma [m]", "ecc [-]", "inc [deg]", "area [m^2]",
            "mass [kg]", "mjd [-]"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df


def _read_debris(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_population(intact_csv: str | Path, debris_csv: str | Path,
                    rng: np.random.Generator, pmd_success: float) -> Population:
    """Build the initial population (intact objects + MASTER debris)."""
    it = _read_intact(Path(intact_csv))
    db = _read_debris(Path(debris_csv))

    cat_it = np.array([classify(t) for t in it["object type"]], dtype=int)
    n_it, n_db = len(it), len(db)

    pop = Population(
        a=np.concatenate([it["sma [m]"].to_numpy(float), db["sma [m]"].to_numpy(float)]),
        e=np.concatenate([it["ecc [-]"].to_numpy(float), db["ecc [-]"].to_numpy(float)]),
        inc=np.concatenate([it["inc [deg]"].to_numpy(float), db["inc [deg]"].to_numpy(float)]),
        area=np.concatenate([it["area [m^2]"].to_numpy(float), db["area [m^2]"].to_numpy(float)]),
        mass=np.concatenate([it["mass [kg]"].to_numpy(float), db["mass [kg]"].to_numpy(float)]),
        launch_mjd=np.concatenate([it["mjd [-]"].to_numpy(float), np.full(n_db, -np.inf)]),
        cat=np.concatenate([cat_it, np.full(n_db, OLD_DEBRIS)]),
        group=np.concatenate([it["object type"].astype(str).to_numpy(),
                              np.full(n_db, "Debris", dtype=object)]).astype(object),
        pmd_ok=rng.random(n_it + n_db) < pmd_success,
        pmd_done=np.zeros(n_it + n_db, dtype=bool),
        alive=np.ones(n_it + n_db, dtype=bool),
    )

    # sanitise: tiny areas/masses, unphysical eccentricities, already-decayed debris
    pop.area = np.clip(pop.area, 1e-4, None)
    pop.mass = np.clip(pop.mass, 0.05, None)
    pop.e = np.clip(pop.e, 0.0, 0.95)
    pop.alive &= pop.perigee_km > REENTRY_PERIGEE_KM
    return pop
