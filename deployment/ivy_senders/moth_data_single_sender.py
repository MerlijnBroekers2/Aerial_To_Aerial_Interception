import time
import numpy as np
import sys
from datetime import datetime
import csv
import math

# USER CONFIG
AC_ID = 42
RATE_HZ = 100
START_DELAY = 0.0

# Per-axis scale for BOTH position and velocity (NED): (sx, sy, sz)
SCALE_NED = (1.0, 1.0, 1.0)

# Fixed offset to add to target POSITION (NED); velocity is NOT offset
POSITION_OFFSET_NED = (0.0, 0.0, -2.0)

# Trials / spoofing
NUM_TRIALS = 10
SPOOF_RADIUS_M = 1.0  # XY radius around (0,0, offset_z)
SPOOF_DURATION_S = 3.0  # seconds to hold the spoof point
RNG_SEED = None  # set an int for reproducible spoof points
SEND_WINDOW_S = 6.0  # max seconds to stream moth data per trial

# Ivy bus
IVY_BUS = "127.255.255.255:2010"

# Optional: print when close enough (no respawn)
PRINT_INTERCEPTION = False
INTERCEPTION_RADIUS = 0.15  # meters

# PAPARAZZI PYTHON PATHS (adjust if needed)
sys.path.insert(0, "/home/merlijn/paparazzi/sw/ext/pprzlink/lib/v1.0/python")
sys.path.insert(1, "/home/merlijn/paparazzi/var/lib/python")

from pprzlink.message import PprzMessage
from pprzlink.ivy import IvyMessagesInterface
from src.models.evaders.moth_evader import MothEvader
from src.utils.config import CONFIG


# IVY BRIDGE (NED)
class IvyBridge:
    """
    - Subscribes to INS messages to keep current drone NED position
    - Sends TARGET_INFO messages with NED position/velocity
      (NOTE: fields are named enu_*, but we intentionally put NED into them)
    """

    def __init__(self, ac_id: int, ivy_bus: str):
        self.ac_id = int(ac_id)
        self.ivy = IvyMessagesInterface(
            agent_name="MothMultiTrialSender",
            start_ivy=True,
            verbose=False,
            ivy_bus=ivy_bus,
        )
        self.drone_pos_ned = None

        # INS raw -> NED scaling (fixed-point to meters)
        self._ins_scaling = {"ins_x": 0.0039063, "ins_y": 0.0039063, "ins_z": 0.0039063}

        # Subscribe to everything; extract INS when present
        self.ivy.subscribe(self._on_msg, regex_or_msg="(.*)")

    def _on_msg(self, ac_id, msg):
        ned_vals = []
        for f, v in zip(msg.fieldnames, msg.fieldvalues):
            if f in self._ins_scaling:
                ned_vals.append(float(v) * self._ins_scaling[f])
        if len(ned_vals) == 3:
            # Keep as NED
            self.drone_pos_ned = np.array(ned_vals, dtype=float)

    def send_target_ned(self, pos_ned: np.ndarray, vel_ned: np.ndarray):
        # Intentionally place NED into enu_* fields
        msg = PprzMessage("datalink", "TARGET_INFO")
        msg["enu_x"], msg["enu_y"], msg["enu_z"] = pos_ned  # NED here on purpose
        msg["enu_xd"], msg["enu_yd"], msg["enu_zd"] = vel_ned  # NED here on purpose
        msg["ac_id"] = self.ac_id
        self.ivy.send(msg)

    def shutdown(self):
        try:
            self.ivy.shutdown()
        except Exception:
            pass


# HELPERS FOR SPOOFING
def random_xy_on_circle(radius: float, rng: np.random.Generator) -> np.ndarray:
    """Return a 2D vector (dx, dy) uniformly on a circle of given radius."""
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return np.array([radius * math.cos(theta), radius * math.sin(theta)], dtype=float)


def send_constant_for_seconds(
    ivy: IvyBridge, pos_ned: np.ndarray, seconds: float, rate_hz: float
):
    """Send a constant target (zero velocity) for 'seconds' seconds at rate_hz."""
    dt = 1.0 / float(rate_hz)
    end_time = time.time() + float(seconds)
    zero_v = np.zeros(3, dtype=float)
    while time.time() < end_time:
        ivy.send_target_ned(pos_ned, zero_v)
        time.sleep(dt)


# MAIN LOOP (NED)
def run():
    if START_DELAY > 0.0:
        time.sleep(START_DELAY)

    ivy = IvyBridge(AC_ID, IVY_BUS)
    rng = np.random.default_rng(RNG_SEED)

    # Ensure proper shapes and types
    scale = np.asarray(SCALE_NED, dtype=float).reshape(3)
    offset = np.asarray(POSITION_OFFSET_NED, dtype=float).reshape(3)
    dt = 1.0 / float(RATE_HZ)

    # Precompute and log post-transform bounds (for info only; uses first file)
    try:
        moth_tmp = MothEvader(config=CONFIG)
        all_pos = np.asarray(moth_tmp.filtered_positions, dtype=float)  # NED
        pos_scaled_offset = all_pos * scale + offset
        pmin = pos_scaled_offset.min(axis=0)
        pmax = pos_scaled_offset.max(axis=0)
        print("[INFO] Position NED ranges after scale+offset (meters):")
        print(f"       N: [{pmin[0]:.3f}, {pmax[0]:.3f}]")
        print(f"       E: [{pmin[1]:.3f}, {pmax[1]:.3f}]")
        print(f"       D: [{pmin[2]:.3f}, {pmax[2]:.3f}]")
        del moth_tmp
    except Exception as e:
        print(f"[WARN] Could not compute position ranges: {e}")

    print(
        f"[INFO] RATE_HZ={RATE_HZ}, SCALE_NED={tuple(scale)}, OFFSET_NED={tuple(offset)}"
    )
    print(
        f"[INFO] Trials={NUM_TRIALS}, Spoof radius={SPOOF_RADIUS_M} m, Spoof duration={SPOOF_DURATION_S} s"
    )

    # Base start point (keep Z from offset fixed)
    base_start_ned = np.array([0.0, 0.0, offset[2]], dtype=float)

    # Keep a log of spoof start points (and times) for later analysis
    spoof_log = []  # rows: dict(trial, x, y, z, t_spoof_start_unix, t_spoof_end_unix)

    try:
        last_pos_ned = None

        for trial_idx in range(1, NUM_TRIALS + 1):
            # 1) Choose random XY point on a 1 m circle around base start (Z unchanged)
            dxy = random_xy_on_circle(SPOOF_RADIUS_M, rng)
            spoof_pos_ned = base_start_ned + np.array(
                [dxy[0], dxy[1], 0.0], dtype=float
            )

            # 2) Send spoof target for ~5 seconds (constant pos, zero vel)
            t_spoof_start = time.time()
            send_constant_for_seconds(ivy, spoof_pos_ned, SPOOF_DURATION_S, RATE_HZ)
            t_spoof_end = time.time()

            spoof_log.append(
                {
                    "trial": trial_idx,
                    "x": float(spoof_pos_ned[0]),
                    "y": float(spoof_pos_ned[1]),
                    "z": float(spoof_pos_ned[2]),
                    "t_spoof_start_unix": t_spoof_start,
                    "t_spoof_end_unix": t_spoof_end,
                }
            )
            print(
                f"[TRIAL {trial_idx}] Spoof at NED=({spoof_pos_ned[0]:.3f}, {spoof_pos_ned[1]:.3f}, {spoof_pos_ned[2]:.3f}) for {SPOOF_DURATION_S:.1f}s"
            )

            # 3) Start moth run from the beginning (fresh instance per trial)
            moth = MothEvader(config=CONFIG)

            t0 = time.time()
            finished = False
            last_pos_ned = None
            last_vel_ned = None
            printed_end = False

            # Respect the earlier of the moth's own end_time (if any) and SEND_WINDOW_S
            if hasattr(moth, "end_time") and moth.end_time is not None:
                send_window = min(float(moth.end_time), float(SEND_WINDOW_S))
            else:
                send_window = float(SEND_WINDOW_S)

            while True:
                if not finished:
                    # Determine if we've reached the configured end
                    t_now = time.time() - t0
                    if t_now >= send_window:
                        finished = True
                        if last_pos_ned is None:
                            # If somehow no step happened, fallback to current moth state
                            state = moth.get_state()
                            last_pos_ned = (
                                scale * np.asarray(state["true_position"], dtype=float)
                                + offset
                            )
                            last_vel_ned = np.zeros(3, dtype=float)
                    else:
                        # Advance and send current state
                        moth.step()
                        state = moth.get_state()  # returns NED
                        pos_ned = (
                            scale * np.asarray(state["true_position"], dtype=float)
                            + offset
                        )
                        vel_ned = scale * np.asarray(state["velocity"], dtype=float)

                        last_pos_ned = pos_ned
                        last_vel_ned = vel_ned
                        ivy.send_target_ned(pos_ned, vel_ned)

                        # Optional proximity print (no respawn)
                        if PRINT_INTERCEPTION and ivy.drone_pos_ned is not None:
                            d = float(np.linalg.norm(ivy.drone_pos_ned - pos_ned))
                            if d < INTERCEPTION_RADIUS:
                                print(f"[INTERCEPTION] d={d:.3f} m (no respawn)")

                else:
                    # Trajectory finished for this trial: hold last position briefly then break
                    if not printed_end:
                        print(
                            f"[TRIAL {trial_idx}] Sending window ended (~{send_window:.1f}s). Holding hover position briefly."
                        )
                        printed_end = True
                    ivy.send_target_ned(
                        np.array([0.0, 0.0, -2.0]),
                        np.zeros(3),
                    )
                    # hold for a short moment between trial segments to be neat
                    time.sleep(0.5)
                    break

                time.sleep(dt)

        # After all trials, hold last position
        print("[INFO] All trials complete. Holding last position.")
        while True:
            hold_pos = last_pos_ned if last_pos_ned is not None else base_start_ned
            ivy.send_target_ned(hold_pos, np.zeros(3))
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting on user interrupt.")
    finally:
        # Save spoof log to CSV for analysis
        try:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fname = f"spoof_starts_{ts}.csv"
            with open(fname, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "trial",
                        "x_n",
                        "y_e",
                        "z_d",
                        "t_spoof_start_unix",
                        "t_spoof_end_unix",
                    ]
                )
                for row in spoof_log:
                    writer.writerow(
                        [
                            row["trial"],
                            row["x"],
                            row["y"],
                            row["z"],
                            row["t_spoof_start_unix"],
                            row["t_spoof_end_unix"],
                        ]
                    )
            print(f"[INFO] Saved spoof start log to {fname}")
        except Exception as e:
            print(f"[WARN] Failed to save spoof start log: {e}")
        ivy.shutdown()


if __name__ == "__main__":
    run()
