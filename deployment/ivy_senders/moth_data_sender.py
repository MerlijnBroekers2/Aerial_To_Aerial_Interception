import sys
import time
import numpy as np
import glob
import os
from pathlib import Path

# Paparazzi Python paths (insert only if they exist)
PPRZ_CANDIDATES = [
    # "/Users/merlijnbroekers/Desktop/paparazzi/sw/ext/pprzlink/lib/v1.0/python",
    # "/Users/merlijnbroekers/Desktop/paparazzi/var/lib/python",
    "/home/merlijn/paparazzi/sw/ext/pprzlink/lib/v1.0/python",
    "/home/merlijn/paparazzi/var/lib/python",
]
for p in reversed(PPRZ_CANDIDATES):
    if os.path.exists(p):
        sys.path.insert(0, p)

from pprzlink.message import PprzMessage
from pprzlink.ivy import IvyMessagesInterface
from src.models.evaders.moth_evader import MothEvader  # NED positions/velocities

# ----------------------------------------------------------------------------
# USER CONFIG (single place to tweak; no CLI)  — EVERYTHING IS NED
# ----------------------------------------------------------------------------
CONFIG = {
    "MOTH_FOLDER": "/home/merlijn/Desktop/Drone_Interception/opogona_moth_data/top_moths/ten_random",
    # Streaming
    "AC_ID": 11,
    "RATE_HZ": 100,
    "START_DELAY": 0.0,
    "IVY_BUS": "127.255.255.255:2010",
    "USE_REAL_GPS": False,  # False: INS (NED); True: REMOTE_GPS_LOCAL (ENU fields)
    # Hard geofence (NED). Your typical D is negative: e.g. D ∈ [-3.5, -1.0]
    "BOUNDS_N": (-2.5, 2.5),
    "BOUNDS_E": (-2.5, 2.5),
    "BOUNDS_D": (-3.0, -1.0),
    # Scaling strategy
    "UNIFORM_SCALE": True,  # True: single factor; False: per-axis fill
    "SAFETY_MARGIN_M": 1e-4,  # shave edges to avoid fence-touching
    # Spawn variety (anti spawn-camp) in N/E plane (NED)
    "MIN_SPAWN_SEP_NE": 0.8,  # m separation between successive spawns (N,E)
    "SPAWN_RESAMPLE_ATTEMPTS": 64,
    "RNG_SEED": None,  # int for reproducible spawns; None → random
    # Optional extra fixed offset (NED) applied after planning scale+offset
    "STATIC_POSITION_OFFSET_NED": (0.0, 0.0, 0.0),
    # Diagnostics
    "PRINT_INTERCEPTION": True,
    "INTERCEPTION_RADIUS": 0.15,  # meters (NED 3D distance)
    # Mode
    "RUN_MODE": "stream",  # "stream" or "preview"
    "HOLD_AFTER_STREAM": True,
    "TRAIL_SECONDS": 5.0,  # trail history to display
    "PLOT_UPDATE_HZ": 50,  # plot refresh rate
    "SHOW_BOUNDS": True,
    "SHOW_INTERCEPTION_RING": True,
    "SHOW_VELOCITY_VECTORS": False,
}


import threading
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class RealTimePlotter:
    def __init__(self, handler):
        self.h = handler
        self.dt_refresh = 1.0 / float(CONFIG["PLOT_UPDATE_HZ"])
        self.trail_seconds = float(CONFIG["TRAIL_SECONDS"])
        self.stop_event = threading.Event()

        # Camera / rotation
        self.rotate_deg_per_sec = 4.0  # nice and slow
        self.base_elev = 25.0  # degrees elevation

        # History: (time, N, E, D)
        self.t0 = time.time()
        self.hist_ev = collections.deque()
        self.hist_dr = collections.deque()

        # Figure and a single 3D axes
        self.fig = plt.figure(figsize=(9, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("3D NED — Evader (red) / Drone (blue)")
        self.ax.set_xlabel("North N (m)")
        self.ax.set_ylabel("East E (m)")
        self.ax.set_zlabel("Down D (m)")

        # Geofence limits (+2 m margin)
        n_min, n_max = CONFIG["BOUNDS_N"]
        e_min, e_max = CONFIG["BOUNDS_E"]
        d_min, d_max = CONFIG["BOUNDS_D"]
        margin = 2.0
        self.lims = (
            (n_min - margin, n_max + margin),
            (e_min - margin, e_max + margin),
            (d_min - margin, d_max + margin),
        )
        self._apply_limits()

        # Draw geofence box (wireframe)
        if CONFIG["SHOW_BOUNDS"]:
            self._draw_box_wireframe(n_min, n_max, e_min, e_max, d_min, d_max)

        # Trails (3D lines)
        (self.ev_line_3d,) = self.ax.plot(
            [], [], [], color="red", linewidth=2, label="Evader"
        )
        (self.dr_line_3d,) = self.ax.plot(
            [], [], [], color="blue", linewidth=2, label="Drone"
        )

        # Heads (as 3D one-point lines with marker)
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
        (nx0, nx1), (ex0, ex1), (dx0, dx1) = self.lims
        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ex0, ex1)
        self.ax.set_zlim(dx1, dx0)

    def _draw_box_wireframe(self, n_min, n_max, e_min, e_max, d_min, d_max):
        # 12 edges of the box
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

        # Get trails
        t_e, N_e, E_e, D_e = self._split(self.hist_ev)
        t_d, N_d, E_d, D_d = self._split(self.hist_dr)

        # Update 3D lines
        self.ev_line_3d.set_data_3d(N_e, E_e, D_e)
        self.dr_line_3d.set_data_3d(N_d, E_d, D_d)

        # Update heads
        if N_e.size:
            self.ev_head_3d.set_data_3d([N_e[-1]], [E_e[-1]], [D_e[-1]])
        if N_d.size:
            self.dr_head_3d.set_data_3d([N_d[-1]], [E_d[-1]], [D_d[-1]])

        # Keep limits fixed to the padded geofence
        self._apply_limits()

        # Slow rotation
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


# class RealTimePlotter:
#     def __init__(self, handler):
#         self.h = handler
#         self.dt_refresh = 1.0 / float(CONFIG["PLOT_UPDATE_HZ"])
#         self.trail_seconds = float(CONFIG["TRAIL_SECONDS"])
#         self.stop_event = threading.Event()

#         # History: (time, N, E, D) — deques keep last TRAIL_SECONDS
#         self.t0 = time.time()
#         self.hist_ev = collections.deque()  # evader: tuples
#         self.hist_dr = collections.deque()  # drone: tuples

#         # Matplotlib setup: top-down (N,E) and D vs time
#         self.fig = plt.figure(figsize=(9, 8))
#         self.ax_ne = self.fig.add_subplot(211)
#         self.ax_d = self.fig.add_subplot(212, sharex=None)

#         self.ax_ne.set_title("Top-down (N vs E) — NED")
#         self.ax_ne.set_xlabel("North N (m)")
#         self.ax_ne.set_ylabel("East E (m)")
#         self.ax_ne.set_aspect("equal", adjustable="box")

#         self.ax_d.set_title("Down D vs Time — NED")
#         self.ax_d.set_xlabel("Time (s)")
#         self.ax_d.set_ylabel("Down D (m)")

#         # Geofence box
#         if CONFIG["SHOW_BOUNDS"]:
#             n_min, n_max = CONFIG["BOUNDS_N"]
#             e_min, e_max = CONFIG["BOUNDS_E"]
#             self.ax_ne.plot(
#                 [n_min, n_max, n_max, n_min, n_min],
#                 [e_min, e_min, e_max, e_max, e_min],
#                 linewidth=2,
#                 linestyle="-",
#                 alpha=0.5,
#             )

#         # Lines/markers for trails and heads
#         (self.ev_line_ne,) = self.ax_ne.plot(
#             [], [], color="red", linewidth=2, label="Evader"
#         )
#         (self.dr_line_ne,) = self.ax_ne.plot(
#             [], [], color="blue", linewidth=2, label="Drone"
#         )
#         self.ev_head_ne = self.ax_ne.plot(
#             [], [], color="red", marker="o", markersize=6
#         )[0]
#         self.dr_head_ne = self.ax_ne.plot(
#             [], [], color="blue", marker="o", markersize=6
#         )[0]

#         (self.ev_line_d,) = self.ax_d.plot(
#             [], [], color="red", linewidth=2, label="Evader D"
#         )
#         (self.dr_line_d,) = self.ax_d.plot(
#             [], [], color="blue", linewidth=2, label="Drone D"
#         )

#         # Optional velocity arrows
#         self.ev_quiv = None
#         self.dr_quiv = None

#         # Interception radius (draw around evader)
#         self.intercept_circle = None
#         if CONFIG["SHOW_INTERCEPTION_RING"]:
#             from matplotlib.patches import Circle

#             self.intercept_circle = Circle(
#                 (0.0, 0.0),
#                 CONFIG["INTERCEPTION_RADIUS"],
#                 fill=False,
#                 linestyle="--",
#                 alpha=0.6,
#             )
#             self.ax_ne.add_patch(self.intercept_circle)

#         self.ax_ne.legend(loc="upper right")
#         self.ax_d.legend(loc="best")

#         self.ani = FuncAnimation(
#             self.fig, self._on_timer, interval=int(1000 * self.dt_refresh), blit=False
#         )

#         self.fig.canvas.mpl_connect("close_event", self._on_close)

#     def _now(self):
#         return time.time() - self.t0

#     def _trim_deque(self, deq):
#         t_now = self._now()
#         # keep last TRAIL_SECONDS
#         while deq and (t_now - deq[0][0] > self.trail_seconds):
#             deq.popleft()

#     def _update_histories(self):
#         t = self._now()
#         # Evader from handler
#         if self.h._target_pos_ned is not None:
#             N, E, D = map(float, self.h._target_pos_ned[:3])
#             self.hist_ev.append((t, N, E, D))

#         # Drone from handler
#         if self.h._drone_pos_ned is not None:
#             N, E, D = map(float, self.h._drone_pos_ned[:3])
#             self.hist_dr.append((t, N, E, D))

#         self._trim_deque(self.hist_ev)
#         self._trim_deque(self.hist_dr)

#     def _on_timer(self, _frame):
#         if self.stop_event.is_set():
#             plt.close(self.fig)
#             return

#         self._update_histories()

#         # Extract arrays
#         def split(deq):
#             if not deq:
#                 return np.array([]), np.array([]), np.array([]), np.array([])
#             arr = np.asarray(deq, dtype=float)
#             return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]  # t, N, E, D

#         t_e, N_e, E_e, D_e = split(self.hist_ev)
#         t_d, N_d, E_d, D_d = split(self.hist_dr)

#         # Update NE trails and heads
#         self.ev_line_ne.set_data(N_e, E_e)
#         self.dr_line_ne.set_data(N_d, E_d)
#         if N_e.size:
#             self.ev_head_ne.set_data([N_e[-1]], [E_e[-1]])
#             if self.intercept_circle is not None:
#                 self.intercept_circle.set_center((N_e[-1], E_e[-1]))
#         if N_d.size:
#             self.dr_head_ne.set_data([N_d[-1]], [E_d[-1]])

#         # Keep NE limits adaptive to bounds or trail
#         if CONFIG["SHOW_BOUNDS"]:
#             n_min, n_max = CONFIG["BOUNDS_N"]
#             e_min, e_max = CONFIG["BOUNDS_E"]
#             self.ax_ne.set_xlim(n_min - 2.0, n_max + 2.0)
#             self.ax_ne.set_ylim(e_min - 2.0, e_max + 2.0)
#         else:
#             # autoscale to trail with margin
#             all_N = (
#                 np.concatenate([N_e, N_d]) if N_e.size or N_d.size else np.array([0.0])
#             )
#             all_E = (
#                 np.concatenate([E_e, E_d]) if E_e.size or E_d.size else np.array([0.0])
#             )
#             if all_N.size:
#                 marginN = max(0.1, 0.05 * (all_N.max() - all_N.min() + 1e-6))
#                 marginE = max(0.1, 0.05 * (all_E.max() - all_E.min() + 1e-6))
#                 self.ax_ne.set_xlim(all_N.min() - marginN, all_N.max() + marginN)
#                 self.ax_ne.set_ylim(all_E.min() - marginE, all_E.max() + marginE)

#         # Update D vs time
#         self.ev_line_d.set_data(t_e, D_e)
#         self.dr_line_d.set_data(t_d, D_d)
#         # Time window is last TRAIL_SECONDS
#         t_now = self._now()
#         self.ax_d.set_xlim(max(0.0, t_now - self.trail_seconds), t_now + 0.01)
#         # Y limits auto
#         all_D = np.concatenate([D_e, D_d]) if D_e.size or D_d.size else np.array([0.0])
#         d_margin = max(0.1, 0.1 * (all_D.max() - all_D.min() + 1e-6))
#         self.ax_d.set_ylim(all_D.min() - d_margin, all_D.max() + d_margin)

#         # Optional velocity vectors (simple frame-to-frame estimates)
#         if CONFIG["SHOW_VELOCITY_VECTORS"]:
#             # Clear old quivers
#             if self.ev_quiv is not None:
#                 self.ev_quiv.remove()
#                 self.ev_quiv = None
#             if self.dr_quiv is not None:
#                 self.dr_quiv.remove()
#                 self.dr_quiv = None

#             def quiv(ax, t, X, Y):
#                 if t.size >= 2:
#                     dt = t[-1] - t[-2]
#                     if dt > 1e-6:
#                         vx = (X[-1] - X[-2]) / dt
#                         vy = (Y[-1] - Y[-2]) / dt
#                         return ax.quiver(
#                             X[-1],
#                             Y[-1],
#                             vx,
#                             vy,
#                             angles="xy",
#                             scale_units="xy",
#                             scale=1.0,
#                             width=0.005,
#                         )
#                 return None

#             self.ev_quiv = quiv(self.ax_ne, t_e, N_e, E_e)
#             self.dr_quiv = quiv(self.ax_ne, t_d, N_d, E_d)

#         # Light on CPU: don’t call tight_layout every frame; just once
#         return []

#     def _on_close(self, _evt):
#         self.stop_event.set()

#     def show(self):
#         self.fig.tight_layout()
#         plt.show()

#     def stop(self):
#         self.stop_event.set()


# Shared Ivy interface
ivy_interface = IvyMessagesInterface(
    agent_name="MothCombinedNED",
    start_ivy=True,
    verbose=False,
    ivy_bus=CONFIG["IVY_BUS"],
)


# ----------------------------------------------------------------------------
# Wrapper that loads each CSV through MothEvader (NED)
# ----------------------------------------------------------------------------
class MothEvaderForSending:
    def __init__(
        self,
        csv_path,
        noise_std=0.0,
        filter_type="passthrough",
        filter_params=None,
        dt=0.01,
    ):
        if filter_params is None:
            filter_params = {}
        self.config = {
            "EVADER": {
                "CSV_FILE": csv_path,
                "NOISE_STD": noise_std,
                "FILTER_TYPE": "ekf",
                "FILTER_PARAMS": {
                    "process_noise": 1e-4,
                    "measurement_noise": 1e-2,
                },
            },
            "DT": dt,
        }
        self.evader = MothEvader(config=self.config)  # NED internally
        self.filtered_positions = np.asarray(
            self.evader.filtered_positions, dtype=float
        )  # NED
        self.end_time = float(self.evader.times[-1])

    def step(self):
        self.evader.step()

    def reset(self):
        self.evader.reset()

    def get_state(self):
        # state["filtered_position"], state["filtered_velocity"] are NED
        return self.evader.get_state()


# ----------------------------------------------------------------------------
# Handler / Sender / Receiver — keep everything in NED
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
        # Stuff NED into enu_* fields on purpose (your convention)
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
        # INS is NED fixed-point → meters
        self.ins_scaling = {"ins_x": 0.0039063, "ins_y": 0.0039063, "ins_z": 0.0039063}
        self.ivy.subscribe(self.handle_message, regex_or_msg="(.*)")

    def handle_message(self, ac_id, msg):
        if not CONFIG["USE_REAL_GPS"] and msg.name == "INS":
            # Keep as NED
            ned_vals = []
            for f, v in zip(msg.fieldnames, msg.fieldvalues):
                if f in self.ins_scaling:
                    ned_vals.append(float(v) * self.ins_scaling[f])
            if len(ned_vals) == 3:
                self.handler.set_drone_pos_ned(np.array(ned_vals, dtype=float))

        elif CONFIG["USE_REAL_GPS"] and msg.name == "REMOTE_GPS_LOCAL":
            # Message fields are ENU; convert to NED: (N=E_enu, E=N_enu, D=-Z_enu)
            enu = {}
            for f, v in zip(msg.fieldnames, msg.fieldvalues):
                if f in ["enu_x", "enu_y", "enu_z"]:
                    enu[f] = float(v)
            if len(enu) == 3:
                ned = np.array([enu["enu_y"], enu["enu_x"], -enu["enu_z"]], dtype=float)
                self.handler.set_drone_pos_ned(ned)


# ----------------------------------------------------------------------------
# Geofence planning in NED: per-evader scale + offset with spawn diversity
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

    # Compute scale (uniform or per-axis), in N/E/D independently if allowed
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

    # Final hard check: entire trajectory inside bounds
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
# Loading evaders via MothEvader wrapper
# ----------------------------------------------------------------------------
def load_moth_evaders_from_folder(folder_path: str, dt: float) -> list:
    paths = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not paths:
        print(f"[ERROR] No moth CSV files found in {folder_path}")
        return []
    evaders = []
    for p in paths:
        try:
            evaders.append(MothEvaderForSending(p, noise_std=0.0, dt=dt))
            print(f"[INFO] Loaded {os.path.basename(p)}")
        except Exception as e:
            print(f"[WARNING] Skipping {os.path.basename(p)}: {e}")
    print(f"[INFO] Loaded {len(evaders)} evader(s) from {folder_path}")
    return evaders


# ----------------------------------------------------------------------------
# Streaming loop (apply per-evader NED transform; keep arrays immutable)
# ----------------------------------------------------------------------------
def run_moth_sender(
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
        st = ev.get_state()  # NED
        pos_ned = np.asarray(st["true_position"], dtype=float)
        vel_ned = np.asarray(st["velocity"], dtype=float)

        pos_out_ned = pos_ned * tf["scale"] + tf["offset"] + extra_offset
        vel_out_ned = vel_ned * tf["scale"]

        handler.set_target_pos_ned(pos_out_ned)
        sender.send_ned(pos_out_ned, vel_out_ned)

        # if handler._drone_pos_ned is None:
        #     print("[INFO] Waiting for INS/REMOTE_GPS_LOCAL…")

        if CONFIG["PRINT_INTERCEPTION"]:
            handler.check_interception()

        time.sleep(dt)


# ----------------------------------------------------------------------------
# Preview / test: visualize N vs E and D vs time (NED)
# ----------------------------------------------------------------------------
def preview_trials(evaders: list, transforms: list):
    import matplotlib.pyplot as plt

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
    _receiver = MothDataReceiver(handler)  # keeps drone NED state

    dt = 1.0 / float(CONFIG["RATE_HZ"])
    folder = Path(CONFIG["MOTH_FOLDER"]).expanduser().resolve()
    evaders = load_moth_evaders_from_folder(str(folder), dt=dt)
    if not evaders:
        return

    transforms = plan_transforms_for_evaders(evaders)

    mode = CONFIG["RUN_MODE"].strip().lower()
    if mode == "preview":
        preview_trials(evaders, transforms)
    elif mode == "stream":
        plotter = RealTimePlotter(handler)
        stop_event = plotter.stop_event

        # Run the sender in a background thread
        worker = threading.Thread(
            target=run_moth_sender,
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
