import time
import numpy as np
import sys
from datetime import datetime
import csv
import math
import os

# USER CONFIG
AC_ID = 42
RATE_HZ = 100
START_DELAY = 0.0

# Per-axis scale for both position and velocity when SENDING (NED)
SCALE_NED = (1.0, 1.0, 1.0) # ! DO not change scaling otherwise this will not work anymore
# Safety bounds for the EVADER in NED (world coords).
# Set any of these to None to disable the check entirely.
EVADER_BOUNDS_N = (-5.0,  5.0)   # meters North
EVADER_BOUNDS_E = (-5.0,  5.0)   # meters East
EVADER_BOUNDS_D = (-3.0, -1.0)   # meters Down (typically negative)


# Fixed offset to add to EVADER POSITION before sending (NED)
# IMPORTANT: this SHIFTED evader position is ALSO used for Δp in the obs.
POSITION_OFFSET_NED = (0.0, 0.0, -2.0)

# Trials / spoofing (optional pre-hold before each RL run)
NUM_TRIALS = 10
SPOOF_RADIUS_M = 1.0  # XY radius around (0,0, offset_z)
SPOOF_DURATION_S = 3.0  # seconds to hold the spoof point
RNG_SEED = None  # set an int for reproducible spoof points
SEND_WINDOW_S = 20.0  # max seconds to stream RL evader per trial

# Ivy bus
IVY_BUS = "127.255.255.255:2010"

# Optional: print when close enough (no respawn)
PRINT_INTERCEPTION = False
INTERCEPTION_RADIUS = 0.15  # meters

# Paparazzi Python paths (insert only if they exist)
PPRZ_CANDIDATES = [
    "/Users/merlijnbroekers/Desktop/paparazzi/sw/ext/pprzlink/lib/v1.0/python",
    "/Users/merlijnbroekers/Desktop/paparazzi/var/lib/python",
    # "/home/merlijn/paparazzi/sw/ext/pprzlink/lib/v1.0/python",
    # "/home/merlijn/paparazzi/var/lib/python",
]
for p in reversed(PPRZ_CANDIDATES):
    if os.path.exists(p):
        sys.path.insert(0, p)

from pprzlink.message import PprzMessage
from pprzlink.ivy import IvyMessagesInterface

from stable_baselines3 import PPO
from src.utils.config import CONFIG

# Shared Ivy interface (no IvyBridge)
ivy_interface = IvyMessagesInterface(
    agent_name="RLEvaderSender",
    start_ivy=True,
    verbose=False,
    ivy_bus=IVY_BUS,
)

def _evader_bounds_enabled():
    return (EVADER_BOUNDS_N is not None and
            EVADER_BOUNDS_E is not None and
            EVADER_BOUNDS_D is not None)

def _in_evader_bounds_ned(p):
    """p: iterable of length 3 (N,E,D) in world NED."""
    if not _evader_bounds_enabled():
        return True
    n_ok = EVADER_BOUNDS_N[0] <= float(p[0]) <= EVADER_BOUNDS_N[1]
    e_ok = EVADER_BOUNDS_E[0] <= float(p[1]) <= EVADER_BOUNDS_E[1]
    d_ok = EVADER_BOUNDS_D[0] <= float(p[2]) <= EVADER_BOUNDS_D[1]
    return n_ok and e_ok and d_ok

def _clamp_to_evader_bounds_ned(p):
    """Clamp p=(N,E,D) to the configured bounds (no-op if disabled)."""
    if not _evader_bounds_enabled():
        return np.asarray(p, dtype=float)
    return np.array([
        np.clip(p[0], EVADER_BOUNDS_N[0], EVADER_BOUNDS_N[1]),
        np.clip(p[1], EVADER_BOUNDS_E[0], EVADER_BOUNDS_E[1]),
        np.clip(p[2], EVADER_BOUNDS_D[0], EVADER_BOUNDS_D[1]),
    ], dtype=float)


# Sender: stuff NED into enu_* (your convention)
def send_target_ned(pos_ned, vel_ned):
    msg = PprzMessage("datalink", "TARGET_INFO")
    msg["enu_x"], msg["enu_y"], msg["enu_z"]   = float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2])
    msg["enu_xd"], msg["enu_yd"], msg["enu_zd"] = float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2])
    msg["ac_id"] = AC_ID
    ivy_interface.send(msg)
    # record for plotting
    if reactive_handler is not None:
        reactive_handler.set_target_pos_ned(pos_ned)


# ---- realtime handler to share state with the plotter ----
class ReactiveHandler:
    def __init__(self):
        self._target_pos_ned = None  # evader (what you send)
        self._drone_pos_ned = None   # pursuer (from INS)

    def set_target_pos_ned(self, pos_ned):
        self._target_pos_ned = np.asarray(pos_ned, dtype=float)

    def set_drone_pos_ned(self, pos_ned):
        self._drone_pos_ned = np.asarray(pos_ned, dtype=float)

# global so send_target_ned can record sends without refactoring every call
reactive_handler = None

import threading, collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ReactiveRealTimePlotter:
    def __init__(self, handler, update_hz=50, trail_seconds=5.0, title="Reactive Evader — NED"):
        self.h = handler
        self.dt_refresh = 1.0 / float(update_hz)
        self.trail_seconds = float(trail_seconds)
        self.stop_event = threading.Event()
        self.margin = 2.0

        # history: (t, N, E, D)
        self.t0 = time.time()
        self.hist_ev = collections.deque()
        self.hist_dr = collections.deque()

        # figure
        self.fig = plt.figure(figsize=(9, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(title)
        self.ax.set_xlabel("North N (m)")
        self.ax.set_ylabel("East E (m)")
        self.ax.set_zlabel("Down D (m)")  # NED

        # lines + heads
        (self.ev_line_3d,) = self.ax.plot([], [], [], linewidth=2, label="Evader", color="red")
        (self.dr_line_3d,) = self.ax.plot([], [], [], linewidth=2, label="Drone",  color="blue")
        (self.ev_head_3d,) = self.ax.plot([], [], [], marker="o", markersize=6, linestyle="None", color="red")
        (self.dr_head_3d,) = self.ax.plot([], [], [], marker="o", markersize=6, linestyle="None", color="blue")
        self.ax.legend(loc="upper left")

        # fixed limits if bounds are enabled
        self.fixed_limits = _evader_bounds_enabled()
        if self.fixed_limits:
            n0, n1 = EVADER_BOUNDS_N
            e0, e1 = EVADER_BOUNDS_E
            d0, d1 = EVADER_BOUNDS_D
            # Apply +2 m margins; invert Z limits (NED)
            self.ax.set_xlim(n0 - self.margin, n1 + self.margin)
            self.ax.set_ylim(e0 - self.margin, e1 + self.margin)
            self.ax.set_zlim(d1 + self.margin, d0 - self.margin)
            # Optional: draw a wireframe box at the exact bounds (no margin)
            self._draw_box_wireframe(n0, n1, e0, e1, d0, d1)
        # else: dynamic autoscale fallback (kept for when any bound is None)

        # slow rotation
        self.rotate_deg_per_sec = 4.0
        self.base_elev = 25.0

        self.ani = FuncAnimation(self.fig, self._on_timer, interval=int(1000*self.dt_refresh), blit=False)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _draw_box_wireframe(self, n_min, n_max, e_min, e_max, d_min, d_max):
        N = [n_min, n_max]
        E = [e_min, e_max]
        D = [d_min, d_max]
        edges = [
            ([N[0], N[1]], [E[0], E[0]], [D[0], D[0]]),
            ([N[1], N[1]], [E[0], E[1]], [D[0], D[0]]),
            ([N[1], N[0]], [E[1], E[1]], [D[0], D[0]]),
            ([N[0], N[0]], [E[1], E[0]], [D[0], D[0]]),
            ([N[0], N[1]], [E[0], E[0]], [D[1], D[1]]),
            ([N[1], N[1]], [E[0], E[1]], [D[1], D[1]]),
            ([N[1], N[0]], [E[1], E[1]], [D[1], D[1]]),
            ([N[0], N[0]], [E[1], E[0]], [D[1], D[1]]),
            ([N[0], N[0]], [E[0], E[0]], [D[0], D[1]]),
            ([N[1], N[1]], [E[0], E[0]], [D[0], D[1]]),
            ([N[1], N[1]], [E[1], E[1]], [D[0], D[1]]),
            ([N[0], N[0]], [E[1], E[1]], [D[0], D[1]]),
        ]
        for n, e, d in edges:
            self.ax.plot(n, e, d, color="0.7", linewidth=1, alpha=0.7)

    def _now(self): 
        return time.time() - self.t0

    def _trim(self, q):
        t_now = self._now()
        while q and (t_now - q[0][0] > self.trail_seconds):
            q.popleft()

    def _update_histories(self):
        t = self._now()
        if self.h._target_pos_ned is not None:
            N,E,D = map(float, self.h._target_pos_ned[:3]); self.hist_ev.append((t,N,E,D))
        if self.h._drone_pos_ned is not None:
            N,E,D = map(float, self.h._drone_pos_ned[:3]); self.hist_dr.append((t,N,E,D))
        self._trim(self.hist_ev); self._trim(self.hist_dr)

    @staticmethod
    def _split(q):
        if not q:
            z = np.array([], dtype=float)
            return z,z,z,z
        arr = np.asarray(q, dtype=float)
        return arr[:,0], arr[:,1], arr[:,2], arr[:,3]

    # Old dynamic autoscale (only used when bounds are disabled)
    def _set_dynamic_limits(self, N_e,E_e,D_e, N_d,E_d,D_d):
        allN = np.concatenate([N_e, N_d]) if N_e.size or N_d.size else np.array([0.0])
        allE = np.concatenate([E_e, E_d]) if E_e.size or E_d.size else np.array([0.0])
        allD = np.concatenate([D_e, D_d]) if D_e.size or D_d.size else np.array([0.0])
        mN = max(0.5, 0.1*(allN.max()-allN.min()+1e-6))
        mE = max(0.5, 0.1*(allE.max()-allE.min()+1e-6))
        mD = max(0.5, 0.1*(allD.max()-allD.min()+1e-6))
        self.ax.set_xlim(allN.min()-mN, allN.max()+mN)
        self.ax.set_ylim(allE.min()-mE, allE.max()+mE)
        self.ax.set_zlim(allD.max()+mD, allD.min()-mD)

    def _on_timer(self, _frame):
        if self.stop_event.is_set():
            plt.close(self.fig); return []
        self._update_histories()

        t_e,N_e,E_e,D_e = self._split(self.hist_ev)
        t_d,N_d,E_d,D_d = self._split(self.hist_dr)

        self.ev_line_3d.set_data_3d(N_e, E_e, D_e)
        self.dr_line_3d.set_data_3d(N_d, E_d, D_d)

        if N_e.size: self.ev_head_3d.set_data_3d([N_e[-1]],[E_e[-1]],[D_e[-1]])
        if N_d.size: self.dr_head_3d.set_data_3d([N_d[-1]],[E_d[-1]],[D_d[-1]])

        # Keep limits fixed if bounds exist; otherwise fall back to dynamic autoscale
        if not self.fixed_limits:
            self._set_dynamic_limits(N_e,E_e,D_e, N_d,E_d,D_d)

        az = (self._now()*self.rotate_deg_per_sec) % 360.0
        self.ax.view_init(elev=self.base_elev, azim=az)
        return []

    def _on_close(self, _evt): 
        self.stop_event.set()

    def show(self):
        self.fig.tight_layout()
        plt.show()

def reactive_sender_loop(stop_event, receiver, evader, rng, dt, base_start_ned):
    spoof_log = []
    last_sent_pos = None
    try:
        for trial_idx in range(1, NUM_TRIALS + 1):
            if stop_event.is_set(): break

            evader.reset(pos0=np.zeros(3, dtype=np.float32),
                vel0=np.zeros(3, dtype=np.float32))

            dxy = random_xy_on_circle(SPOOF_RADIUS_M, rng)
            spoof_pos_ned = base_start_ned + np.array([dxy[0], dxy[1], 0.0], dtype=float)

            t_spoof_start = time.time()
            end_time = t_spoof_start + float(SPOOF_DURATION_S)
            zero_v = np.zeros(3)
            while time.time() < end_time and not stop_event.is_set():
                send_target_ned(spoof_pos_ned, zero_v)
                time.sleep(dt)
            t_spoof_end = time.time()
            print(f"[TRIAL {trial_idx}] Spoof at NED=({spoof_pos_ned[0]:.3f}, {spoof_pos_ned[1]:.3f}, {spoof_pos_ned[2]:.3f}) for {SPOOF_DURATION_S:.1f}s")

            spoof_log.append({"trial": trial_idx, "x": float(spoof_pos_ned[0]), "y": float(spoof_pos_ned[1]), "z": float(spoof_pos_ned[2]),
                              "t_spoof_start_unix": t_spoof_start, "t_spoof_end_unix": t_spoof_end})

            t0 = time.time(); printed_end = False
            while not stop_event.is_set():
                # if receiver.drone_pos_ned is None: ### Send origin as pursuer positon until messages are received 
                #     time.sleep(dt); continue

                


                p = receiver.drone_pos_ned
                v = receiver.drone_vel_ned
                send_pos, send_vel = evader.step(p, v)

                # --- safety: terminate trial if evader leaves bounds ---
                if not _in_evader_bounds_ned(send_pos):
                    clipped = _clamp_to_evader_bounds_ned(send_pos)
                    print(
                        "[TRIAL {ti}] Evader out of bounds at NED=({:.3f},{:.3f},{:.3f}) → "
                        "clamping to ({:.3f},{:.3f},{:.3f}) and terminating trial."
                        .format(send_pos[0], send_pos[1], send_pos[2], clipped[0], clipped[1], clipped[2], ti=trial_idx)
                    )
                    # Final settle/hold like normal end condition
                    send_target_ned(clipped, np.zeros(3))
                    last_sent_pos = clipped  # so the post-trial hold is safe/in-bounds
                    time.sleep(0.5)
                    break


                send_target_ned(send_pos, send_vel)
                last_sent_pos = send_pos

                if PRINT_INTERCEPTION:
                    d = float(np.linalg.norm(receiver.drone_pos_ned - evader.pos_world))
                    if d < INTERCEPTION_RADIUS:
                        print(f"[INTERCEPTION] d={d:.3f} m (no respawn)")

                if (time.time() - t0) >= float(SEND_WINDOW_S):
                    if not printed_end:
                        print(f"[TRIAL {trial_idx}] RL send window ended (~{SEND_WINDOW_S:.1f}s). Holding briefly.")
                        printed_end = True
                    send_target_ned(send_pos, np.zeros(3))
                    time.sleep(0.5)
                    break

                time.sleep(dt)

        print("[INFO] Loop ending. Holding last position until stopped.")
        while not stop_event.is_set():
            hold_pos = last_sent_pos if last_sent_pos is not None else base_start_ned
            send_target_ned(hold_pos, np.zeros(3))
            time.sleep(0.2)
    finally:
        return spoof_log



# Receiver: keep pursuer NED pos/vel from INS
class RLDroneReceiver:
    def __init__(self, handler):
        self.h = handler
        self.drone_pos_ned = np.zeros(3, dtype=float)
        self.drone_vel_ned = np.zeros(3, dtype=float)
        self._ins_pos_scale = {"ins_x": 0.0039063, "ins_y": 0.0039063, "ins_z": 0.0039063}
        self._ins_vel_scale = {"ins_xd": 0.0000019, "ins_yd": 0.0000019, "ins_zd": 0.0000019}

        if self.h is not None:
            self.h.set_drone_pos_ned(self.drone_pos_ned)
        ivy_interface.subscribe(self._on_msg, regex_or_msg="(.*)")
        

    def _on_msg(self, ac_id, msg):
        if msg.name != "INS":
            return
        pos_map = {"ins_x": 0, "ins_y": 1, "ins_z": 2}
        vel_map = {"ins_xd": 0, "ins_yd": 1, "ins_zd": 2}
        pos = np.empty(3, dtype=float); vel = np.empty(3, dtype=float)
        got_pos = [False, False, False]; got_vel = [False, False, False]
        for f, v in zip(msg.fieldnames, msg.fieldvalues):
            if f in pos_map:
                i = pos_map[f]; pos[i] = float(v) * self._ins_pos_scale[f]; got_pos[i] = True
            elif f in vel_map:
                i = vel_map[f]; vel[i] = float(v) * self._ins_vel_scale[f]; got_vel[i] = True
        if all(got_pos):
            self.drone_pos_ned = pos
            if self.h is not None:
                self.h.set_drone_pos_ned(pos)
        if all(got_vel):
            self.drone_vel_ned = vel




# IVY BRIDGE (NED)
def __init__(self, ac_id: int, ivy_bus: str):
    self.ac_id = int(ac_id)
    self.ivy = IvyMessagesInterface(
        agent_name="RLEvaderSender",
        start_ivy=True,
        verbose=False,
        ivy_bus=ivy_bus,
    )

    # Latest NED state of the pursuer (drone)
    self.drone_pos_ned = None
    self.drone_vel_ned = np.zeros(3, dtype=float)

    # INS fixed-point → SI scaling
    self._ins_pos_scale = {
        "ins_x": 0.0039063,
        "ins_y": 0.0039063,
        "ins_z": 0.0039063,
    }  # m
    self._ins_vel_scale = {
        "ins_xd": 0.0000019,
        "ins_yd": 0.0000019,
        "ins_zd": 0.0000019,
    }  # m/s

    # Subscribe to everything; extract INS when present
    self.ivy.subscribe(self._on_msg, regex_or_msg="(.*)")

    def _on_msg(self, ac_id, msg):
        # Map fields to indices to avoid order assumptions
        pos_map = {"ins_x": 0, "ins_y": 1, "ins_z": 2}
        vel_map = {"ins_xd": 0, "ins_yd": 1, "ins_zd": 2}

        pos = np.empty(3, dtype=float)
        vel = np.empty(3, dtype=float)
        got_pos = [False, False, False]
        got_vel = [False, False, False]

        for f, v in zip(msg.fieldnames, msg.fieldvalues):
            if f in pos_map:
                i = pos_map[f]
                pos[i] = float(v) * self._ins_pos_scale[f]
                got_pos[i] = True
            elif f in vel_map:
                i = vel_map[f]
                vel[i] = float(v) * self._ins_vel_scale[f]
                got_vel[i] = True

        if all(got_pos):
            self.drone_pos_ned = pos
        if all(got_vel):
            self.drone_vel_ned = vel

    def shutdown(self):
        try:
            self.ivy.shutdown()
        except Exception:
            pass


# RL EVADER RUNTIME (reactive, continuous)
class ReactiveEvaderRunner:
    """
    Minimal real-time wrapper around a PPO evader policy.
    - Internal state (pos, vel) in meters (NED), dt from RATE_HZ.
    - At each tick: build obs = [Δp, Δv, p_e_shifted] and query policy.
    - Apply accel limits, integrate, clamp speed, return shifted/scaled outputs.
    """

    def __init__(
        self, config, dt, offset_ned, scale_ned, model_path=None, deterministic=True
    ):
        self.cfg = config
        ev = self.cfg["EVADER"]
        self.dt = float(dt)
        self.max_accel = float(ev["MAX_ACCEL"])
        self.max_speed = float(ev["MAX_SPEED"])

        self.offset = np.asarray(offset_ned, dtype=float).reshape(3)
        self.scale = np.asarray(scale_ned, dtype=float).reshape(3)

        self.pos = np.zeros(3, dtype=np.float32)  # start at origin (as trained)
        self.vel = np.zeros(3, dtype=np.float32)
        self.acc_last = np.zeros(3, dtype=np.float32)

        path = model_path or ev["RL_MODEL_PATH"]
        self.policy: PPO = PPO.load(path)
        self.deterministic = deterministic
        print(f"[ReactiveEvaderRunner] Loaded PPO policy from: {path}")

    def reset(self, pos0=None, vel0=None):
        self.pos[:] = (
            np.zeros(3, dtype=np.float32)
            if pos0 is None
            else np.asarray(pos0, dtype=np.float32)
        )
        self.vel[:] = (
            np.zeros(3, dtype=np.float32)
            if vel0 is None
            else np.asarray(vel0, dtype=np.float32)
        )
        self.acc_last[:] = 0.0

    # ---- frame transforms ----
    def _to_local_pos(self, p_world):
        return (np.asarray(p_world, dtype=np.float32).reshape(3) - self.offset) / self.scale

    def _to_local_vel(self, v_world):
        return np.asarray(v_world, dtype=np.float32).reshape(3) / self.scale

    def _to_world_pos(self, p_local):
        return (np.asarray(p_local, dtype=np.float32).reshape(3) * self.scale) + self.offset

    def _to_world_vel(self, v_local):
        return np.asarray(v_local, dtype=np.float32).reshape(3) * self.scale

    def step(self, pursuer_pos_ned, pursuer_vel_ned):
        # --- measurements -> model frame ---
        p_p = self._to_local_pos(pursuer_pos_ned)   # pursuer position in model/local coords
        v_p = self._to_local_vel(pursuer_vel_ned)   # pursuer velocity in model/local coords

        # --- evader in model/local coords (already origin-centered) ---
        p_e = self.pos
        v_e = self.vel

        # build obs exactly as trained
        dp  = p_p - p_e
        dv  = v_p - v_e
        obs = np.concatenate([dp, dv, p_e], axis=0).astype(np.float32)

        actions, _ = self.policy.predict(obs, deterministic=self.deterministic)
        acc_cmd = np.clip(np.asarray(actions, dtype=np.float32).reshape(3), -1.0, 1.0) * self.max_accel
        self.acc_last = acc_cmd

        # integrate in model frame
        self.vel += acc_cmd * self.dt
        speed = float(np.linalg.norm(self.vel))
        if speed > self.max_speed:
            self.vel *= self.max_speed / (speed + 1e-8)
        self.pos += self.vel * self.dt

        # --- model -> world for sending ---
        out_pos = self._to_world_pos(self.pos)
        out_vel = self._to_world_vel(self.vel)
        return out_pos.astype(float), out_vel.astype(float)

    @property
    def pos_world(self):
        return self._to_world_pos(self.pos).astype(float)

    @property
    def vel_current(self):
        return self.vel.astype(float)


# HELPERS FOR SPOOFING
def random_xy_on_circle(radius: float, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return np.array([radius * math.cos(theta), radius * math.sin(theta)], dtype=float)


def send_constant_for_seconds(pos_ned: np.ndarray, seconds: float, rate_hz: float):
    dt = 1.0 / float(rate_hz)
    end_time = time.time() + float(seconds)
    zero_v = np.zeros(3, dtype=float)
    while time.time() < end_time:
        send_target_ned(pos_ned, zero_v)
        time.sleep(dt)



def run():
    global reactive_handler

    if START_DELAY > 0.0:
        time.sleep(START_DELAY)

    rng    = np.random.default_rng(RNG_SEED)
    dt     = 1.0 / float(RATE_HZ)
    scale  = np.asarray(SCALE_NED, dtype=float).reshape(3)
    offset = np.asarray(POSITION_OFFSET_NED, dtype=float).reshape(3)

    # handler + receiver + evader
    reactive_handler = ReactiveHandler()
    receiver = RLDroneReceiver(reactive_handler)

    evader = ReactiveEvaderRunner(CONFIG, dt=dt, offset_ned=offset, scale_ned=scale, model_path=None, deterministic=True)
    evader.reset()
    base_start_ned = offset.copy()

    print(f"[INFO] RATE_HZ={RATE_HZ}, SCALE_NED={tuple(scale)}, OFFSET_NED={tuple(offset)}")
    print(f"[INFO] Trials={NUM_TRIALS}, Spoof radius={SPOOF_RADIUS_M} m, Spoof duration={SPOOF_DURATION_S} s")

    # plotter + worker
    plotter = ReactiveRealTimePlotter(reactive_handler, update_hz=50, trail_seconds=5.0)
    stop_event = plotter.stop_event

    worker = threading.Thread(
        target=reactive_sender_loop,
        args=(stop_event, receiver, evader, rng, dt, base_start_ned),
        daemon=True,
    )
    worker.start()

    # show blocks until you close the window; that sets stop_event
    plotter.show()
    worker.join(timeout=1.0)

    try:
        ivy_interface.shutdown()
    except Exception:
        pass



if __name__ == "__main__":
    run()
