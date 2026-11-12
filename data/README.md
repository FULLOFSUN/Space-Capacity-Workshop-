# Input data

Put the exercise input files here, under these exact names:

| File | Used by | What it is |
|---|---|---|
| `ex1_intact_objs_nfl.csv` | `nfl` | Intact objects in LEO today (DISCOS), no future launches |
| `ex1_intact_objs_baseline.csv` | `baseline`, `bonus` | Same population plus the extrapolated launch traffic (SARIMAX non-constellation launches, Starlink and Guowang with replenishment) |
| `ex1_debris_data.csv` | all | MASTER fragments, 10 cm to 1 m |

They come from the ESA/MIT long-term simulation exercise (first exercise of the workshop on
[indico.esa.int](https://indico.esa.int/event/648/)). They are not committed to this repository
(see `.gitignore`); download them from the workshop page.

Expected columns (intact files):

```
object type, sma [m], ecc [-], inc [deg], area [m^2], mass [kg], status [-], RAAN [deg], AoP [deg], M [deg], date [-], MJD [-]
```

`status [-]` is the launch year, and `MJD [-]` is the launch date the simulation uses to add an object.
The debris file has the same orbital and physical columns, without `object type`, `date` or `MJD`.
RAAN, AoP and M are not used: the collision model works with altitude shells.
