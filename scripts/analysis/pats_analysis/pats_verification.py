"""
Animate drone & target positions from PATS CSV.
python scripts/analysis/pats_analysis/animate_drone_target.py --csv /path/to/your_file.csv --save-gif out.gif --speed 5
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------- Column helpers -----------------------------
CAND_DRONE_POS = [["posX_drone"], ["posY_drone"], ["posZ_drone"]]
CAND_TARGET_POS = [
    ["posX_bestinsect"],
    ["posY_bestinsect"],
    ["posZ_bestinsect"],
]
CAND_TIME_ELAPSED = ["elapsed", "time", "timestamp"]
CAND_TIME_DT = ["dt", "delta_t", "delta_ms"]
CAND_ACC_TARGET = ["accX_target", "accY_target", "accZ_target"]
CAND_ACC_COMMANDED = ["accX_commanded", "accY_commanded", "accZ_commanded"]


def _normalize_name(s: str) -> str:
    return s.strip().replace(" ", "").lower()


def pick_column(df: pd.DataFrame, names):
    """Pick the first column present (case/space-insensitive) from names list."""
    norm_map = {_normalize_name(c): c for c in df.columns}
    for name in names:
        if isinstance(name, (list, tuple)):
            for n in name:
                key = _normalize_name(n)
                if key in norm_map:
                    return norm_map[key]
        else:
            key = _normalize_name(name)
            if key in norm_map:
                return norm_map[key]
    return None


# ----------------------------- Coords mapping -----------------------------
def map_coords(v: np.ndarray, mode: str) -> np.ndarray:
    """
    mode='xyz' -> identity
    mode='pats' -> (x, y, z) -> (x, z, y)  (so that 'y up' = original y)
    """
    if mode.lower() == "pats":
        return np.array([v[0], v[2], v[1]], dtype=float)
    return v.astype(float)


# ----------------------------- Main logic --------------------------------
def load_data(csv_path: str, delim: str, coord_mode: str, acc_source: str):
    df = pd.read_csv(csv_path, sep=delim, engine="python", skipinitialspace=True)
    # positions
    pos_d_cols = [pick_column(df, x) for x in CAND_DRONE_POS]
    pos_t_cols = [pick_column(df, x) for x in CAND_TARGET_POS]
    if not all(pos_d_cols) or not all(pos_t_cols):
        raise RuntimeError(
            f"Missing required position columns.\n"
            f"  Drone columns found: {pos_d_cols}\n"
            f"  Target columns found: {pos_t_cols}"
        )

    for c in pos_d_cols + pos_t_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # time
    elapsed_col = pick_column(df, CAND_TIME_ELAPSED)
    dt_col = pick_column(df, CAND_TIME_DT)
    if elapsed_col:
        df[elapsed_col] = pd.to_numeric(df[elapsed_col], errors="coerce")
        times = df[elapsed_col].to_numpy(dtype=float)
        if np.isfinite(times[0]):
            times = times - times[0]
    else:
        if not dt_col:
            # assume fixed step (50 Hz) if no timing at all
            times = np.arange(len(df), dtype=float) * 0.02
        else:
            df[dt_col] = pd.to_numeric(df[dt_col], errors="coerce")
            dt = df[dt_col].to_numpy(dtype=float)
            # Heuristic: if median dt > 10, treat as milliseconds
            med = np.nanmedian(dt) if np.isfinite(dt).any() else 20.0
            if med > 10.0:
                dt = dt / 1000.0
            dt = np.nan_to_num(dt, nan=(med / (1000.0 if med > 10.0 else 1.0)))
            times = np.cumsum(dt)
            times = times - times[0]

    # acceleration (optional)
    acc_cols = None
    if acc_source == "target":
        acc_cols = [pick_column(df, [c]) for c in CAND_ACC_TARGET]
    elif acc_source == "commanded":
        acc_cols = [pick_column(df, [c]) for c in CAND_ACC_COMMANDED]

    acc = None
    if acc_cols and all(acc_cols):
        for c in acc_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        acc = df.loc[:, acc_cols].to_numpy(dtype=float)

    pos_d = df.loc[:, pos_d_cols].to_numpy(dtype=float)
    pos_t = df.loc[:, pos_t_cols].to_numpy(dtype=float)

    # map coords
    pos_d = np.apply_along_axis(map_coords, 1, pos_d, coord_mode)
    pos_t = np.apply_along_axis(map_coords, 1, pos_t, coord_mode)
    if acc is not None:
        acc = np.apply_along_axis(map_coords, 1, acc, coord_mode)

    # valid mask
    mask = (
        np.isfinite(pos_d).all(axis=1)
        & np.isfinite(pos_t).all(axis=1)
        & np.isfinite(times)
    )
    times = times[mask]
    pos_d = pos_d[mask]
    pos_t = pos_t[mask]
    if acc is not None:
        acc = acc[mask]

    if len(times) < 2:
        raise RuntimeError("Not enough valid frames to animate.")

    return times, pos_d, pos_t, acc


def make_animation(
    times: np.ndarray,
    pos_d: np.ndarray,
    pos_t: np.ndarray,
    acc: np.ndarray | None,
    speed: float,
    min_ms: int,
    max_ms: int,
    coord_mode: str,
    acc_scale: float,
    save_gif: str | None,
    save_mp4: str | None,
):
    # variable frame spacing (clamped) with speed factor
    dt = np.diff(times, prepend=times[0])
    intervals_ms = np.clip((dt / max(speed, 1e-6)) * 1000.0, min_ms, max_ms).astype(int)
    if len(intervals_ms):
        intervals_ms[0] = min_ms

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Drone vs Target — positions over time")

    # labels
    if coord_mode.lower() == "pats":
        ax.set_xlabel("X")
        ax.set_ylabel("Z (back)")
        ax.set_zlabel("Y (up)")
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    # artists
    scat_drone = ax.scatter([], [], [], s=50, label="Drone")
    scat_target = ax.scatter([], [], [], s=50, label="Target")
    acc_vec = None
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    # axes limits
    all_pts = np.vstack([pos_d, pos_t])
    mins = np.nanmin(all_pts, axis=0) - 0.5
    maxs = np.nanmax(all_pts, axis=0) + 0.5
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    try:
        spans = maxs - mins
        ax.set_box_aspect((spans[0], spans[1], spans[2]))
    except Exception:
        pass
    ax.legend(loc="lower left")

    def update(i):
        nonlocal acc_vec
        p_d = pos_d[i]
        p_t = pos_t[i]

        scat_drone._offsets3d = ([p_d[0]], [p_d[1]], [p_d[2]])
        scat_target._offsets3d = ([p_t[0]], [p_t[1]], [p_t[2]])

        if acc is not None:
            if acc_vec is not None:
                try:
                    acc_vec.remove()
                except Exception:
                    pass
            a = acc[i] * acc_scale
            acc_vec = ax.quiver(p_d[0], p_d[1], p_d[2], a[0], a[1], a[2])

        time_text.set_text(f"t = {times[i]:.3f} s")
        if i + 1 < len(intervals_ms):
            ani.event_source.interval = int(intervals_ms[i + 1])

        return (
            (scat_drone, scat_target, time_text)
            if acc_vec is None
            else (scat_drone, scat_target, acc_vec, time_text)
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=int(intervals_ms[0]),
        blit=False,
        repeat=True,
    )

    # saving (optional)
    if save_gif:
        try:
            from matplotlib.animation import PillowWriter
        except Exception as e:
            raise RuntimeError(
                "Saving GIF requires 'pillow' (pip install pillow)."
            ) from e
        median_ms = float(np.median(intervals_ms)) or 40.0
        fps = max(1, int(round(1000.0 / median_ms)))
        ani.save(save_gif, writer=PillowWriter(fps=fps))
        print(f"[OK] GIF saved: {save_gif}")

    if save_mp4:
        try:
            ani.save(save_mp4, writer="ffmpeg", dpi=150)
            print(f"[OK] MP4 saved: {save_mp4}")
        except Exception as e:
            print("[WARN] MP4 save failed (need ffmpeg).", e)

    plt.show()


# ----------------------------- CLI ---------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Animate drone & target positions from a CSV log."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file (semicolon-delimited by default).",
    )
    ap.add_argument("--delim", default=";", help="Field delimiter (default=';').")
    ap.add_argument(
        "--coord",
        choices=["xyz", "pats"],
        default="xyz",
        help="Coordinate mapping: 'xyz' (identity) or 'pats' (x,z,y).",
    )
    ap.add_argument(
        "--acc-source",
        choices=["none", "target", "commanded"],
        default="target",
        help="Acceleration arrow source (default: target).",
    )
    ap.add_argument(
        "--acc-scale",
        type=float,
        default=0.5,
        help="Scale for acceleration vector (default 0.5).",
    )
    ap.add_argument(
        "--speed", type=float, default=5.0, help="Playback speed factor (>1 is faster)."
    )
    ap.add_argument(
        "--min-interval-ms", type=int, default=1, help="Min frame interval (ms)."
    )
    ap.add_argument(
        "--max-interval-ms", type=int, default=200, help="Max frame interval (ms)."
    )
    ap.add_argument(
        "--save-gif", default=None, help="Optional path to save GIF (requires pillow)."
    )
    ap.add_argument(
        "--save-mp4", default=None, help="Optional path to save MP4 (requires ffmpeg)."
    )
    args = ap.parse_args()

    times, pos_d, pos_t, acc = load_data(
        csv_path=args.csv,
        delim=args.delim,
        coord_mode=args.coord,
        acc_source=args.acc_source,
    )

    make_animation(
        times=times,
        pos_d=pos_d,
        pos_t=pos_t,
        acc=(None if args.acc_source == "none" else acc),
        speed=args.speed,
        min_ms=args.min_interval_ms,
        max_ms=args.max_interval_ms,
        coord_mode=args.coord,
        acc_scale=args.acc_scale,
        save_gif=args.save_gif,
        save_mp4=args.save_mp4,
    )


if __name__ == "__main__":
    main()
