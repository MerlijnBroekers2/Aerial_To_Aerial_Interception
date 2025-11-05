# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import glob
import os
from pathlib import Path
import threading
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paparazzi Python paths (insert only if they exist)
PPRZ_CANDIDATES = [
    # "/Users/yourname/paparazzi/sw/ext/pprzlink/lib/v1.0/python",
    # "/Users/yourname/paparazzi/var/lib/python",
    "/home/merlijn/paparazzi/sw/ext/pprzlink/lib/v1.0/python",
    "/home/merlijn/paparazzi/var/lib/python",
]
for p in reversed(PPRZ_CANDIDATES):
    if os.path.exists(p):
        sys.path.insert(0, p)

from pprzlink.message import PprzMessage
from pprzlink.ivy import IvyMessagesInterface

# Your evader class
from src.models.evaders.pliska_evader import PliskaEvader  # <- using Pliska!


# ----------------------------------------------------------------------------
# USER CONFIG — EVERYTHING IN NED
# ----------------------------------------------------------------------------
CONFIG = {
    # Where your CSVs live
    "PLISKA_FOLDER": "/home/merlijn/Desktop/Drone_Interception/pliska_csv",
    # Paparazzi / streaming
    "AC_ID": 11,
    "RATE_HZ": 100,
    "START_DELAY": 0.0,
    "IVY_BUS": "127.255.255.255:2010",
    "USE_REAL_GPS": False,  # False: use INS (NED); True: convert REMOTE_GPS_LOCAL (ENU) to NED
    # Geofence (NED). Typical D is negative (z-down): e.g. D ∈ [-3.5, -1.0]
    "BOUNDS_N": (-2.0, 2.0),
    "BOUNDS_E": (-2.0, 2.0),
    "BOUNDS_D": (-2.5, -1.0),
    # Planner
    "SPEED_MULTIPLIER": 7.0,
    "UNIFORM_SCALE": True,  # True: single factor; False: per-axis fill
    "SAFETY_MARGIN_M": 1e-4,  # shave edges to avoid fence-touching
    "MIN_SPAWN_SEP_NE": 0.8,  # m separation between successive spawns (N,E)
    "SPAWN_RESAMPLE_ATTEMPTS": 64,
    "RNG_SEED": None,  # int for reproducible spawns; None → random
    "STATIC_POSITION_OFFSET_NED": (0.0, 0.0, 0.0),
    # Interception
    "PRINT_INTERCEPTION": True,
    "INTERCEPTION_RADIUS": 0.15,  # meters (3D NED distance)
    # Mode
    "RUN_MODE": "stream",  # "stream" or "preview"
    # Plot
    "TRAIL_SECONDS": 5.0,  # trail history length
    "PLOT_UPDATE_HZ": 50,  # GUI refresh rate
    "SHOW_BOUNDS": True,
    "ROTATE_DEG_PER_SEC": 4.0,  # slow rotation around Z
    "BASE_ELEV_DEG": 25.0,  # camera elevation in degrees
    "PADDED_MARGIN_M": 2.0,  # plot limits extend geofence by this margin
    # Pliska evader config
    "PLISKA_NOISE_STD_POS": 0.0,  # meters (noise added in Pliska)
    "PLISKA_POSITION_BOUND": 2.0,  # meters (Pliska pre-scale bound)
    "PLISKA_FILTER_TYPE": "passthrough",  # or "ekf"
    "PLISKA_FILTER_PARAMS": {},  # e.g. {"q_acc": 50., "r_pos": 0.5}
    "PLISKA_EVAL_USE_FILTERED_AS_GT": True,  # use filtered as “true”
}


# ----------------------------------------------------------------------------
# Ivy interface (shared)
# ----------------------------------------------------------------------------
ivy_interface = IvyMessagesInterface(
    agent_name="PliskaNEDStreamer",
    start_ivy=True,
    verbose=False,
    ivy_bus=CONFIG["IVY_BUS"],
)


# ----------------------------------------------------------------------------
# Real-time 3D rotating plotter (NED, z-down)
# ----------------------------------------------------------------------------
class RealTimePlotter3D:
    def __init__(self, handler):
        self.h = handler
        self.dt_refresh = 1.0 / float(CONFIG["PLOT_UPDATE_HZ"])
        self.trail_seconds = float(CONFIG["TRAIL_SECONDS"])
        self.stop_event = threading.Event()

        # Camera / rotation
        self.rotate_deg_per_sec = float(CONFIG["ROTATE_DEG_PER_SEC"])
        self.base_elev = float(CONFIG["BASE_ELEV_DEG"])

        # History: (time, N, E, D)
        self.t0 = time.time()
        self.hist_ev = collections.deque()
        self.hist_dr = collections.deque()

        # Figure: 3D axes
        self.fig = plt.figure(figsize=(9, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("3D NED — Evader (red) / Drone (blue)")
        self.ax.set_xlabel("North N (m)")
        self.ax.set_ylabel("East E (m)")
        self.ax.set_zlabel("Down D (m)")

        # Limits (+margin)
        n_min, n_max = CONFIG["BOUNDS_N"]
        e_min, e_max = CONFIG["BOUNDS_E"]
        d_min, d_max = CONFIG["BOUNDS_D"]
        margin = float(CONFIG["PADDED_MARGIN_M"])
        self.lims = (
            (n_min - margin, n_max + margin),
            (e_min - margin, e_max + margin),
            (d_min - margin, d_max + margin),
        )
        self._apply_limits()

        # Optional geofence wireframe
        if CONFIG["SHOW_BOUNDS"]:
            self._draw_box_wireframe(n_min, n_max, e_min, e_max, d_min, d_max)

        # Trails (3D lines)
        (self.ev_line_3d,) = self.ax.plot(
            [], [], [], color="red", linewidth=2, label="Evader"
        )
        (self.dr_line_3d,) = self.ax.plot(
            [], [], [], color="blue", linewidth=2, label="Drone"
        )

        # Heads (markers)
        (self.ev_head_3d,) = self.ax.plot(
            [], [], [], marker="o", markersize=6, color="red", linestyle="None"
        )
        (self.dr_head_3d,) = self.ax.plot(
            [], [], [], marker="o", markersize=6, color="blue", linestyle="None"
        )

        self.ax.legend(loc="upper left")

        self.ani = FuncAnimation(
            self.fig, self._on_timer, interval=int(1000 * self.dt_refresh), blit=False
        )
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _apply_limits(self):
        # Flip z-axis so up is visually up even though D is negative
        (nx0, nx1), (ex0, ex1), (dx0, dx1) = self.lims
        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ex0, ex1)
        self.ax.set_zlim(dx1, dx0)  # reversed to flip visually (z-up look)

    def _draw_box_wireframe(self, n_min, n_max, e_min, e_max, d_min, d_max):
        N = [n_min, n_max]
        E = [e_min, e_max]
        D = [d_min, d_max]
        edges = [
            # bottom rectangle (at d_min)
            ([N[0], N[1]], [E[0], E[0]], [D[0], D[0]]),
            ([N[1], N[1]], [E[0], E[1]], [D[0], D[0]]),
            ([N[1], N[0]], [E[1], E[1]], [D[0], D[0]]),
            ([N[0], N[0]], [E[1], E[0]], [D[0], D[0]]),
            # top rectangle (at d_max)
            ([N[0], N[1]], [E[0], E[0]], [D[1], D[1]]),
            ([N[1], N[1]], [E[0], E[1]], [D[1], D[1]]),
            ([N[1], N[0]], [E[1], E[1]], [D[1], D[1]]),
            ([N[0], N[0]], [E[1], E[0]], [D[1], D[1]]),
            # verticals
            ([N[0], N[0]], [E[0], E[0]], [D[0], D[1]]),
            ([N[1], N[1]], [E[0], E[0]], [D[0], D[1]]),
            ([N[1], N[1]], [E[1], E[1]], [D[0], D[1]]),
            ([N[0], N[0]], [E[1], E[1]], [D[0], D[1]]),
        ]
        for n, e, d in edges:
            self.ax.plot(n, e, d, color="0.6", linewidth=1, alpha=0.7)

    def _now(self):
        return time.time() - self.t0

    def _trim_deque(self, deq):
        t_now = self._now()
        while deq and (t_now - deq[0][0] > self.trail_seconds):
            deq.popleft()

    def _update_histories(self):
        t = self._now()
        if self.h._target_pos_ned is not None:
            N, E, D = map(float, self.h._target_pos_ned[:3])
            self.hist_ev.append((t, N, E, D))
        if self.h._drone_pos_ned is not None:
            N, E, D = map(float, self.h._drone_pos_ned[:3])
            self.hist_dr.append((t, N, E, D))
        self._trim_deque(self.hist_ev)
        self._trim_deque(self.hist_dr)

    def _split(self, deq):
        if not deq:
            z = np.array([], dtype=float)
            return z, z, z, z
        arr = np.asarray(deq, dtype=float)
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]  # t, N, E, D

    def _on_timer(self, _frame):
        if self.stop_event.is_set():
            plt.close(self.fig)
            return
        self._update_histories()

        # Trails
        _, N_e, E_e, D_e = self._split(self.hist_ev)
        _, N_d, E_d, D_d = self._split(self.hist_dr)

        self.ev_line_3d.set_data_3d(N_e, E_e, D_e)
        self.dr_line_3d.set_data_3d(N_d, E_d, D_d)

        if N_e.size:
            self.ev_head_3d.set_data_3d([N_e[-1]], [E_e[-1]], [D_e[-1]])
        if N_d.size:
            self.dr_head_3d.set_data_3d([N_d[-1]], [E_d[-1]], [D_d[-1]])

        self._apply_limits()

        # Rotate slowly
        az = (self._now() * self.rotate_deg_per_sec) % 360.0
        self.ax.view_init(elev=self.base_elev, azim=az)

        return []

    def _on_close(self, _evt):
        self.stop_event.set()

    def show(self):
        self.fig.tight_layout()
        plt.show()

    def stop(self):
        self.stop_event.set()


# ----------------------------------------------------------------------------
# Pliska wrapper for streaming (keeps interface same as before)
# ----------------------------------------------------------------------------
class PliskaEvaderForSending:
    def __init__(self, csv_path, dt=0.01):
        self.config = {
            "EVADER": {
                "CSV_FILE": csv_path,
                "NOISE_STD_POS": CONFIG["PLISKA_NOISE_STD_POS"],
                "PLISKA_POSITION_BOUND": CONFIG["PLISKA_POSITION_BOUND"],
                "FILTER_TYPE": CONFIG["PLISKA_FILTER_TYPE"],
                "FILTER_PARAMS": CONFIG["PLISKA_FILTER_PARAMS"],
                "EVAL_USE_FILTERED_AS_GT": CONFIG["PLISKA_EVAL_USE_FILTERED_AS_GT"],
                "SPEED_MULTIPLIER": CONFIG["SPEED_MULTIPLIER"],
            },
            "DT": dt,
        }
        self.evader = PliskaEvader(config=self.config)
        self.filtered_positions = np.asarray(
            self.evader.filtered_positions, dtype=float
        )
        self.end_time = float(self.evader.end_time)

    def step(self):
        self.evader.step()

    def reset(self):
        self.evader.reset()

    def get_state(self):
        return self.evader.get_state()


# ----------------------------------------------------------------------------
# Handler / Sender / Receiver — NED throughout
# ----------------------------------------------------------------------------
class MothDataHandler:
    def __init__(self):
        self._target_pos_ned = None
        self._drone_pos_ned = None
        self.intercepted = False

    def set_target_pos_ned(self, pos_ned):
        self._target_pos_ned = np.array(pos_ned, dtype=float)

    def set_drone_pos_ned(self, pos_ned):
        self._drone_pos_ned = np.array(pos_ned, dtype=float)

    def check_interception(self):
        if self._target_pos_ned is not None and self._drone_pos_ned is not None:
            d = float(np.linalg.norm(self._target_pos_ned - self._drone_pos_ned))
            if d < CONFIG["INTERCEPTION_RADIUS"]:
                print(f"[INTERCEPTION] d={d:.3f} m")
                self.intercepted = True


class MothDataSender:
    def __init__(self, handler, ac_id):
        self.ivy = ivy_interface
        self.ac_id = int(ac_id)
        self.handler = handler

    def send_ned(self, pos_ned, vel_ned):
        # Stuff NED into enu_* fields (your convention)
        msg = PprzMessage("datalink", "TARGET_INFO")
        msg["enu_x"], msg["enu_y"], msg["enu_z"] = (
            float(pos_ned[0]),
            float(pos_ned[1]),
            float(pos_ned[2]),
        )
        msg["enu_xd"], msg["enu_yd"], msg["enu_zd"] = (
            float(vel_ned[0]),
            float(vel_ned[1]),
            float(vel_ned[2]),
        )
        msg["ac_id"] = self.ac_id
        self.ivy.send(msg)


class MothDataReceiver:
    def __init__(self, handler):
        self.ivy = ivy_interface
        self.handler = handler
        # INS is NED fixed-point → meters (1/256 m per LSB ≈ 0.00390625)
        self.ins_scaling = {
            "ins_x": 1.0 / 256.0,
            "ins_y": 1.0 / 256.0,
            "ins_z": 1.0 / 256.0,
        }
        self.ivy.subscribe(self.handle_message, regex_or_msg="(.*)")

    def handle_message(self, ac_id, msg):
        if not CONFIG["USE_REAL_GPS"] and msg.name == "INS":
            ned_vals = []
            for f, v in zip(msg.fieldnames, msg.fieldvalues):
                if f in self.ins_scaling:
                    ned_vals.append(float(v) * self.ins_scaling[f])
            if len(ned_vals) == 3:
                self.handler.set_drone_pos_ned(np.array(ned_vals, dtype=float))

        elif CONFIG["USE_REAL_GPS"] and msg.name == "REMOTE_GPS_LOCAL":
            # ENU → NED: (N=E_enu, E=N_enu, D=-Z_enu)
            enu = {}
            for f, v in zip(msg.fieldnames, msg.fieldvalues):
                if f in ["enu_x", "enu_y", "enu_z"]:
                    enu[f] = float(v)
            if len(enu) == 3:
                ned = np.array([enu["enu_y"], enu["enu_x"], -enu["enu_z"]], dtype=float)
                self.handler.set_drone_pos_ned(ned)


# ----------------------------------------------------------------------------
# Geofence planning in NED
# ----------------------------------------------------------------------------
class BoundsNED:
    def __init__(self, n, e, d, safety_margin):
        self.mins = np.array(
            [n[0] + safety_margin, e[0] + safety_margin, d[0] + safety_margin],
            dtype=float,
        )
        self.maxs = np.array(
            [n[1] - safety_margin, e[1] - safety_margin, d[1] - safety_margin],
            dtype=float,
        )
        self.spans = self.maxs - self.mins


def _safe_span(v: float) -> float:
    return float(v) if abs(v) > 1e-12 else 0.0


def _plan_scale_offset_for_one_ned(
    traj_ned: np.ndarray,
    bounds: BoundsNED,
    rng: np.random.Generator,
    used_spawns_ne: list,
    uniform_scale: bool,
    min_spawn_sep_ne: float,
    resample_attempts: int,
) -> tuple:
    """Return (scale[3], offset[3], spawn_ned[3]) keeping full path inside NED bounds."""
    pmin = traj_ned.min(axis=0)
    pmax = traj_ned.max(axis=0)
    spans0 = pmax - pmin
    center0 = 0.5 * (pmin + pmax)

    if uniform_scale:
        denom = np.array([_safe_span(spans0[i]) for i in range(3)], dtype=float)
        factors = np.where(denom > 0.0, bounds.spans / denom, np.inf)
        s = np.min(factors[np.isfinite(factors)]) if np.isfinite(factors).any() else 1.0
        scale = np.array([s, s, s], dtype=float)
    else:
        scale = np.array(
            [
                (bounds.spans[0] / spans0[0]) if _safe_span(spans0[0]) > 0 else 1.0,
                (bounds.spans[1] / spans0[1]) if _safe_span(spans0[1]) > 0 else 1.0,
                (bounds.spans[2] / spans0[2]) if _safe_span(spans0[2]) > 0 else 1.0,
            ],
            dtype=float,
        )

    scaled_spans = spans0 * scale
    low_c = bounds.mins + 0.5 * scaled_spans
    high_c = bounds.maxs - 0.5 * scaled_spans
    low_c = np.minimum(low_c, high_c)  # guard

    best = None
    farthest = -1.0
    for _ in range(resample_attempts):
        center = rng.uniform(low=low_c, high=high_c)
        offset = center - scale * center0
        spawn = scale * traj_ned[0] + offset  # NED at t=0

        # Enforce spawn separation in N/E
        ok_sep = True
        for prev in used_spawns_ne:
            if np.linalg.norm(spawn[:2] - prev[:2]) < min_spawn_sep_ne:
                ok_sep = False
                break
        if not ok_sep:
            continue

        if used_spawns_ne:
            dsep = float(np.linalg.norm(spawn[:2] - used_spawns_ne[-1][:2]))
        else:
            dsep = 1e9
        if dsep > farthest:
            farthest = dsep
            best = (scale.copy(), offset.copy(), spawn.copy())
            if dsep >= 1.5 * min_spawn_sep_ne:
                break

    if best is None:
        center = 0.5 * (low_c + high_c)
        offset = center - scale * center0
        spawn = scale * traj_ned[0] + offset
        best = (scale.copy(), offset.copy(), spawn.copy())

    # Final check
    pos_full = traj_ned * best[0][None, :] + best[1][None, :]
    if (pos_full < bounds.mins[None, :] - 1e-9).any() or (
        pos_full > bounds.maxs[None, :] + 1e-9
    ).any():
        raise RuntimeError("Planned transform violates geofence (unexpected).")

    return best


def plan_transforms_for_evaders(evaders: list) -> list:
    """Compute (scale, offset, spawn_ned) for each evader in NED."""
    cfg = CONFIG
    bounds = BoundsNED(
        cfg["BOUNDS_N"], cfg["BOUNDS_E"], cfg["BOUNDS_D"], cfg["SAFETY_MARGIN_M"]
    )
    rng = np.random.default_rng(cfg["RNG_SEED"])
    used_spawns_ne = []
    results = []

    for idx, ev in enumerate(evaders, 1):
        traj_ned = np.asarray(ev.filtered_positions, dtype=float)  # NED
        scale, offset, spawn = _plan_scale_offset_for_one_ned(
            traj_ned=traj_ned,
            bounds=bounds,
            rng=rng,
            used_spawns_ne=used_spawns_ne,
            uniform_scale=cfg["UNIFORM_SCALE"],
            min_spawn_sep_ne=cfg["MIN_SPAWN_SEP_NE"],
            resample_attempts=cfg["SPAWN_RESAMPLE_ATTEMPTS"],
        )
        used_spawns_ne.append(spawn[:2].copy())

        pos_so = traj_ned * scale + offset
        pmin = pos_so.min(axis=0)
        pmax = pos_so.max(axis=0)
        print(
            f"[PLAN {idx}] scale={tuple(np.round(scale,5))}  offset={tuple(np.round(offset,5))}  spawn(NED)={tuple(np.round(spawn,5))}"
        )
        print(
            f"[PLAN {idx}] N:[{pmin[0]:.3f},{pmax[0]:.3f}]  E:[{pmin[1]:.3f},{pmax[1]:.3f}]  D:[{pmin[2]:.3f},{pmax[2]:.3f}]"
        )

        results.append({"scale": scale, "offset": offset, "spawn_ned": spawn})

    return results


# ----------------------------------------------------------------------------
# Load Pliska evaders from folder
# ----------------------------------------------------------------------------
def load_pliska_evaders_from_folder(folder_path: str, dt: float) -> list:
    paths = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not paths:
        print(f"[ERROR] No CSV files found in {folder_path}")
        return []
    evaders = []
    for p in paths:
        try:
            evaders.append(PliskaEvaderForSending(p, dt=dt))
            print(f"[INFO] Loaded {os.path.basename(p)}")
        except Exception as e:
            print(f"[WARNING] Skipping {os.path.basename(p)}: {e}")
    print(f"[INFO] Loaded {len(evaders)} evader(s) from {folder_path}")
    return evaders


# ----------------------------------------------------------------------------
# Streaming loop: applies per-evader NED transform and sends to Paparazzi
# ----------------------------------------------------------------------------
def run_pliska_sender(
    handler, evaders: list, transforms: list, stop_event: threading.Event
):
    sender = MothDataSender(handler, CONFIG["AC_ID"])
    dt = 1.0 / float(CONFIG["RATE_HZ"])
    extra_offset = np.asarray(CONFIG["STATIC_POSITION_OFFSET_NED"], dtype=float)

    if CONFIG["START_DELAY"] > 0.0:
        time.sleep(CONFIG["START_DELAY"])

    ev_idx = 0
    ev = evaders[ev_idx]
    tf = transforms[ev_idx]
    ev.reset()
    t0 = time.time()

    while not stop_event.is_set():
        t_query = time.time() - t0 - CONFIG["START_DELAY"]
        if handler.intercepted or t_query > ev.end_time:
            ev_idx = (ev_idx + 1) % len(evaders)
            ev = evaders[ev_idx]
            tf = transforms[ev_idx]
            ev.reset()
            t0 = time.time()
            handler.intercepted = False
            print(f"[INFO] Switched to evader {ev_idx+1}.")
            time.sleep(0.1)
            continue

        ev.step()
        st = ev.get_state()  # dict with "true_position", "filtered_velocity", etc.

        # Choose what you want to stream as target:
        pos_ned = np.asarray(st["true_position"], dtype=float)
        vel_ned = np.asarray(st["velocity"], dtype=float)

        # Apply per-evader affine transform, then optional extra NED offset
        pos_out_ned = pos_ned * tf["scale"] + tf["offset"] + extra_offset
        vel_out_ned = vel_ned * tf["scale"]  # scale vel component-wise

        handler.set_target_pos_ned(pos_out_ned)
        sender.send_ned(pos_out_ned, vel_out_ned)

        if CONFIG["PRINT_INTERCEPTION"]:
            handler.check_interception()

        time.sleep(dt)


# ----------------------------------------------------------------------------
# Preview: quick static 2D projections to sanity check bounds (optional)
# ----------------------------------------------------------------------------
def preview_trials(evaders: list, transforms: list):
    n_min, n_max = CONFIG["BOUNDS_N"]
    e_min, e_max = CONFIG["BOUNDS_E"]
    d_min, d_max = CONFIG["BOUNDS_D"]
    extra_offset = np.asarray(CONFIG["STATIC_POSITION_OFFSET_NED"], dtype=float)

    fig_ne = plt.figure()
    ax_ne = fig_ne.add_subplot(111)
    ax_ne.plot(
        [n_min, n_max, n_max, n_min, n_min],
        [e_min, e_min, e_max, e_max, e_min],
        linewidth=2,
    )
    ax_ne.set_aspect("equal", adjustable="box")
    ax_ne.set_xlabel("North N (m)")
    ax_ne.set_ylabel("East E (m)")
    ax_ne.set_title("Top-down preview (N vs E) within geofence (NED)")

    fig_d = plt.figure()
    ax_d = fig_d.add_subplot(111)
    ax_d.axhline(d_min, linestyle="--")
    ax_d.axhline(d_max, linestyle="--")
    ax_d.set_xlabel("Sample index")
    ax_d.set_ylabel("Down D (m)")
    ax_d.set_title("Down (D) vs time per trial (NED)")

    for idx, (ev, tf) in enumerate(zip(evaders, transforms), 1):
        pos_ned = np.asarray(ev.filtered_positions, dtype=float)
        pos_so = pos_ned * tf["scale"] + tf["offset"] + extra_offset
        ax_ne.plot(pos_so[:, 0], pos_so[:, 1], label=f"trial {idx}")
        ax_ne.plot([pos_so[0, 0]], [pos_so[0, 1]], marker="o")  # spawn marker
        ax_d.plot(np.arange(pos_so.shape[0]), pos_so[:, 2], label=f"trial {idx}")

    ax_ne.legend()
    ax_d.legend()
    plt.show()


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    handler = MothDataHandler()
    _receiver = MothDataReceiver(handler)  # keeps drone NED state up-to-date

    dt = 1.0 / float(CONFIG["RATE_HZ"])
    folder = Path(CONFIG["PLISKA_FOLDER"]).expanduser().resolve()
    evaders = load_pliska_evaders_from_folder(str(folder), dt=dt)
    if not evaders:
        return

    transforms = plan_transforms_for_evaders(evaders)

    mode = CONFIG["RUN_MODE"].strip().lower()
    if mode == "preview":
        preview_trials(evaders, transforms)
    elif mode == "stream":
        plotter = RealTimePlotter3D(handler)
        stop_event = plotter.stop_event

        # Run the sender in a background thread
        worker = threading.Thread(
            target=run_pliska_sender,
            args=(handler, evaders, transforms, stop_event),
            daemon=True,
        )
        worker.start()

        # Show the GUI (blocks until window closed)
        plotter.show()

        # Window closed → stop sender and exit
        stop_event.set()
        worker.join(timeout=1.0)
        print("[INFO] Stream/plot shut down.")
    else:
        raise ValueError("CONFIG['RUN_MODE'] must be 'preview' or 'stream'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Program stopped by user.")
