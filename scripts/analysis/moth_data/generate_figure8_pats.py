"""
This script was made to quickly generate a figure 8 mock insect that can be used for testing on the PATS
system. The generated .csv can be replayed as an actual insect on the PATS section, leading to the PATS-X
tracking/intercepting the figure 8 flying drone

Take a real flight log to ensure the validity check and other pheripherals are in place. Then overwrites the
positional and velocity data with a figure 8 trajectory
"""

import math
import pandas as pd
import numpy as np

# ===================== USER CONFIG =====================
INPUT_CSV = (
    "/Users/merlijnbroekers/Desktop/Drone_Interception/008.csv"  # <- real insect CSV
)
OUTPUT_CSV = "out_figure8_small_long.csv"  # <- new CSV to write

# Figure-8 size (max excursion about OFFSET), in meters (PATS frame)
AMP_X = 0.1  # left/right   (PATS x)
AMP_Y = 0.1  # up/down      (PATS y)
AMP_Z = 0.5  # back/forward (PATS z)

# Offset of origin in PATS (meters)
OFFSET_X, OFFSET_Y, OFFSET_Z = (0.0, -1.00, -1.50)

# Speed control: choose ONE
TARGET_PEAK_SPEED_MPS = 1.0  # set to None to use CYCLE_HZ
CYCLE_HZ = 0.25  # used when TARGET_PEAK_SPEED_MPS is None

# If True, force *_valid flags to 1 for pos/vel/acc
FORCE_VALID_FLAGS = True

# Columns to overwrite (everything else is preserved)
POS_COLS = [
    "posX_insect",
    "posY_insect",
    "posZ_insect",
    "sposX_insect",
    "sposY_insect",
    "sposZ_insect",
]
VEL_COLS = ["svelX_insect", "svelY_insect", "svelZ_insect"]
ACC_COLS = ["saccX_insect", "saccY_insect", "saccZ_insect"]
VALID_COLS = ["pos_valid_insect", "vel_valid_insect", "acc_valid_insect"]


def figure8_state(t, Ax, Ay, Az, ox, oy, oz, omega):
    """
    x = Ax * sin(tau)
    z = 0.5 * Az * sin(2tau)
    y = Ay * sin(tau)
    tau = omega * t
    """
    tau = omega * t
    s1, c1 = math.sin(tau), math.cos(tau)
    s2, c2 = math.sin(2.0 * tau), math.cos(2.0 * tau)

    # position about origin
    x = Ax * s1
    y = Ay * s1
    z = 0.5 * Az * s2

    # velocity (analytic)
    vx = omega * Ax * c1
    vy = omega * Ay * c1
    vz = omega * Az * c2

    # acceleration (analytic)
    ax = -(omega**2) * Ax * s1
    ay = -(omega**2) * Ay * s1
    az = -2.0 * (omega**2) * Az * s2

    return (x + ox, y + oy, z + oz, vx, vy, vz, ax, ay, az)


def choose_omega_from_peak_speed(Ax, Ay, Az, v_peak):
    denom = math.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    if denom < 1e-9:
        return 0.0
    return max(v_peak, 1e-6) / denom


def main():
    # Read original CSV (semicolon separated), preserve column order
    df = pd.read_csv(INPUT_CSV, sep=";", engine="python", dtype=str)

    # Ensure elapsed exists and is numeric; build relative time t = elapsed - elapsed0
    if "elapsed" not in df.columns:
        raise RuntimeError("Input CSV missing 'elapsed' column.")
    elapsed = pd.to_numeric(df["elapsed"], errors="coerce")
    if elapsed.isna().all():
        raise RuntimeError("All 'elapsed' values are NaN; cannot time the trajectory.")
    t0 = elapsed.dropna().iloc[0]
    t_rel = (elapsed - t0).astype(float).fillna(method="ffill").fillna(0.0).to_numpy()

    # Determine omega
    if TARGET_PEAK_SPEED_MPS is not None:
        omega = choose_omega_from_peak_speed(AMP_X, AMP_Y, AMP_Z, TARGET_PEAK_SPEED_MPS)
    else:
        omega = 2.0 * math.pi * CYCLE_HZ

    states = np.array(
        [
            figure8_state(t, AMP_X, AMP_Y, AMP_Z, OFFSET_X, OFFSET_Y, OFFSET_Z, omega)
            for t in t_rel
        ]
    )  # shape (N, 9)

    # Unpack
    pos = states[:, 0:3]
    vel = states[:, 3:6]
    acc = states[:, 6:9]

    for c in POS_COLS + VEL_COLS + ACC_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Overwrite only the positional/velocity/acceleration data
    df.loc[:, ["posX_insect", "posY_insect", "posZ_insect"]] = pos
    df.loc[:, ["sposX_insect", "sposY_insect", "sposZ_insect"]] = pos
    df.loc[:, ["svelX_insect", "svelY_insect", "svelZ_insect"]] = vel
    df.loc[:, ["saccX_insect", "saccY_insect", "saccZ_insect"]] = acc

    # Optionally force validity flags
    if FORCE_VALID_FLAGS:
        for c in VALID_COLS:
            if c in df.columns:
                df[c] = 1

    # Write output (semicolon-separated), keep column order
    df.to_csv(OUTPUT_CSV, sep=";", index=False, float_format="%.6f")
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
    cyc_hz = omega / (2.0 * math.pi)
    approx_vmax = omega * math.sqrt(AMP_X**2 + AMP_Y**2 + AMP_Z**2)
    print(
        f"Figure-8: cycle ≈ {cyc_hz:.3f} Hz, approx peak speed ≈ {approx_vmax:.3f} m/s"
    )


if __name__ == "__main__":
    main()
