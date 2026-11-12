"""Physical constants and default model parameters."""

MU = 3.986004418e14          # Earth gravitational parameter [m^3/s^2]
RE = 6378137.0               # Earth equatorial radius [m]
SEC_PER_YEAR = 365.25 * 86400.0
MJD_J2026 = 61041.0          # 2026-01-01 00:00 UTC as Modified Julian Date

# Population bookkeeping
REENTRY_PERIGEE_KM = 120.0   # objects whose perigee drops below this are removed
REENTRY_MEAN_ALT_KM = 150.0  # ...or whose mean altitude drops below this
LEO_TOP_KM = 2000.0          # upper bound of the collision shells

# Collision model
SHELL_WIDTH_KM = 50.0        # width of the spherical shells used for collision rates
V_REL_KM_S = 10.0            # mean relative impact velocity in LEO [km/s]
CATASTROPHIC_EMR_J_PER_G = 40.0

# NASA Standard Breakup Model
LC_MIN_M = 0.10              # smallest fragment tracked (matches the 10 cm input debris)
CD = 2.2                     # drag coefficient for all objects

# Operations
MISSION_LIFETIME_YR = 5.0
