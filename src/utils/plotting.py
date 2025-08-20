# utils/plotting.py
# --------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
from matplotlib import cm, colors
from sklearn.cluster import DBSCAN


# --------------------------------------------------------------------------
def animate_pursuit_evasion(
    sim_result: dict,
    *,
    pos_offset: np.ndarray | None = None,  # identical to real-log viewer
    scale_factor: float = 1.0,
    speed_factor: int = 1,
    interval: int = 50,
    save_file: str | None = None,
):
    """
    Visualise a single pursuit–evasion run recorded via Simulation.record_state().

    Parameters
    ----------
    pos_offset   : optional ENU translation (3,) applied to both trajectories
    scale_factor : multiply all positions – useful for matching real-world units
    speed_factor : skip n-1 frames between draws
    interval     : delay between frames [ms]
    save_file    : "*.gif" → Pillow, anything else → ffmpeg
    """

    # ------------------------------------------------------------------ data
    H = sim_result["history"]
    N = len(H)

    t = np.array([h["time"] for h in H])
    pursuer_pos = np.stack([h["p_state"]["true_position"] for h in H])
    evader_pos = np.stack([h["e_state"]["filtered_position"] for h in H])
    attitude = np.stack([h["p_state"]["attitude"] for h in H])
    v_pursuer = np.stack([h["p_state"]["velocity"] for h in H])
    v_evader = np.stack([h["e_state"]["filtered_velocity"] for h in H])
    a_cmd = np.stack([h["p_state"]["acc_command"] for h in H])

    # optional offset / scale (keeps interface identical to animate_pn_log)
    if pos_offset is not None:
        pursuer_pos += pos_offset
        evader_pos += pos_offset
    pursuer_pos *= scale_factor
    evader_pos *= scale_factor

    # -------------------------- possible interceptions ---------------------
    if sim_result.get("interceptions"):  # list may be empty
        hit_times = np.array([hit["time"] for hit in sim_result["interceptions"]])
        hit_points = np.stack(
            [hit["pursuer_pos"] for hit in sim_result["interceptions"]]
        )
    else:
        hit_times = np.empty(0)
        hit_points = np.empty((0, 3))

    # ------------------------------------------------------------------ fig
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Simulation run RL_CTBR")

    xyz_all = np.vstack([pursuer_pos, evader_pos])
    pad = 0.5
    ax.set_xlim(xyz_all[:, 0].min() - pad, xyz_all[:, 0].max() + pad)
    ax.set_ylim(xyz_all[:, 1].min() - pad, xyz_all[:, 1].max() + pad)
    ax.set_zlim(xyz_all[:, 2].min() - pad, xyz_all[:, 2].max() + pad)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # ground “floor”
    floor = np.array(
        [
            [xyz_all[:, 0].min() - pad, xyz_all[:, 1].min() - pad, 0],
            [xyz_all[:, 0].min() - pad, xyz_all[:, 1].max() + pad, 0],
            [xyz_all[:, 0].max() + pad, xyz_all[:, 1].max() + pad, 0],
            [xyz_all[:, 0].max() + pad, xyz_all[:, 1].min() - pad, 0],
        ]
    )
    ax.add_collection3d(Poly3DCollection([floor], color="gray", alpha=0.2))

    # trails & markers
    (p_line,) = ax.plot([], [], [], lw=2, color="red")
    (e_line,) = ax.plot([], [], [], lw=2, color="blue")
    (p_mark,) = ax.plot([], [], [], "o", color="red")
    (e_mark,) = ax.plot([], [], [], "o", color="blue")

    # tiny triangular “airframe”
    L, W = 0.4, 0.4
    body_shape = np.array([[L, -L / 2, -L / 2], [0, W / 2, -W / 2], [0, 0, 0]])
    drone_poly = Poly3DCollection([body_shape.T], color="green", alpha=0.8)
    ax.add_collection3d(drone_poly)

    # arrow segments – simple Line3D objects (cheap to update)
    accel_line = Line3D([], [], [], color="magenta")
    vel_line = Line3D([], [], [], color="cyan")
    tgt_line = Line3D([], [], [], color="yellow")
    ax.add_line(accel_line)
    ax.add_line(vel_line)
    ax.add_line(tgt_line)

    # interception scatter (single collection, starts empty)
    hit_scatter = ax.scatter([], [], [], marker="x", s=80, c="black")

    time_txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # ---------------------------------------------------------------- helpers
    def _rot(phi, theta, psi):
        c, s = np.cos, np.sin
        Rz = np.array([[c(psi), -s(psi), 0], [s(psi), c(psi), 0], [0, 0, 1]])
        Ry = np.array([[c(theta), 0, s(theta)], [0, 1, 0], [-s(theta), 0, c(theta)]])
        Rx = np.array([[1, 0, 0], [0, c(phi), -s(phi)], [0, s(phi), c(phi)]])
        return Rz @ Ry @ Rx

    trail_window = 3.0  # seconds of trail to show

    # ---------------------------- init ------------------------------------
    def init():
        for ln in (p_line, e_line, accel_line, vel_line, tgt_line):
            ln.set_data_3d([], [], [])
        for mk in (p_mark, e_mark):
            mk.set_data_3d([], [], [])
        drone_poly.set_verts([body_shape.T])
        hit_scatter._offsets3d = ([], [], [])
        time_txt.set_text("")
        return (
            p_line,
            e_line,
            p_mark,
            e_mark,
            drone_poly,
            accel_line,
            vel_line,
            tgt_line,
            hit_scatter,
            time_txt,
        )

    # ---------------------------- update ----------------------------------
    def update(frame_idx):
        i = min(frame_idx, N - 1)

        # orientation
        phi, theta, psi = attitude[i]
        R = _rot(phi, theta, psi)
        drone_poly.set_verts([(R @ body_shape).T + pursuer_pos[i]])

        # trails (3-s sliding window)
        start = np.searchsorted(t, t[i] - trail_window)
        p_line.set_data_3d(
            pursuer_pos[start : i + 1, 0],
            pursuer_pos[start : i + 1, 1],
            pursuer_pos[start : i + 1, 2],
        )
        e_line.set_data_3d(
            evader_pos[start : i + 1, 0],
            evader_pos[start : i + 1, 1],
            evader_pos[start : i + 1, 2],
        )

        # point markers (wrap scalars in a list)
        p_mark.set_data_3d(
            [pursuer_pos[i, 0]], [pursuer_pos[i, 1]], [pursuer_pos[i, 2]]
        )
        e_mark.set_data_3d([evader_pos[i, 0]], [evader_pos[i, 1]], [evader_pos[i, 2]])

        # arrow segments
        a_dir = a_cmd[i] / (np.linalg.norm(a_cmd[i]) + 1e-9)
        a_world = R @ a_dir
        accel_line.set_data_3d(
            [pursuer_pos[i, 0], pursuer_pos[i, 0] + a_world[0]],
            [pursuer_pos[i, 1], pursuer_pos[i, 1] + a_world[1]],
            [pursuer_pos[i, 2], pursuer_pos[i, 2] + a_world[2]],
        )
        vel_line.set_data_3d(
            [pursuer_pos[i, 0], pursuer_pos[i, 0] + v_pursuer[i, 0]],
            [pursuer_pos[i, 1], pursuer_pos[i, 1] + v_pursuer[i, 1]],
            [pursuer_pos[i, 2], pursuer_pos[i, 2] + v_pursuer[i, 2]],
        )
        tgt_line.set_data_3d(
            [evader_pos[i, 0], evader_pos[i, 0] + v_evader[i, 0]],
            [evader_pos[i, 1], evader_pos[i, 1] + v_evader[i, 1]],
            [evader_pos[i, 2], evader_pos[i, 2] + v_evader[i, 2]],
        )

        # reveal interceptions whose time ≤ current frame time
        if hit_times.size:
            recent = (hit_times <= t[i]) & (hit_times >= t[i] - trail_window)
            if recent.any():
                pts = hit_points[recent]
                hit_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            else:  # nothing “recent” → hide them
                hit_scatter._offsets3d = ([], [], [])

        time_txt.set_text(f"time : {t[i]:.2f} s")
        return (
            p_line,
            e_line,
            p_mark,
            e_mark,
            drone_poly,
            accel_line,
            vel_line,
            tgt_line,
            hit_scatter,
            time_txt,
        )

    # ---------------------------------------------------------------- animation
    frames = range(0, N, speed_factor)
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval,
        blit=False,
        repeat=False,
    )

    # legend (matches real-log viewer)
    ax.legend(
        handles=[
            Line2D([0], [0], color="red", lw=2, label="Pursuer"),
            Line2D([0], [0], color="blue", lw=2, label="Evader"),
            Line2D([0], [0], color="magenta", lw=2, label="Accel cmd"),
            Line2D([0], [0], color="cyan", lw=2, label="Pursuer vel"),
            Line2D([0], [0], color="yellow", lw=2, label="Evader vel"),
        ],
        loc="upper right",
    )

    # optional export
    if save_file:
        writer = (
            PillowWriter(fps=60) if save_file.endswith(".gif") else FFMpegWriter(fps=60)
        )
        anim.save(save_file, writer=writer)

    plt.show()
    return anim
