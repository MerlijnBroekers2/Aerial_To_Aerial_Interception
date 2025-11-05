from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation


def animate_pn_log(
    pn_csv_path,
    start_send_time,
    pos_offset,
    scale_factor,
    speed_factor=1,
    interval=50,
    save_file=None,
):
    df = pd.read_csv(pn_csv_path)

    coords = [
        "pos_x",
        "pos_y",
        "pos_z",
        "pos_target_x",
        "pos_target_y",
        "pos_target_z",
        "accel_command_x",
        "accel_command_y",
        "accel_command_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "vel_target_x",
        "vel_target_y",
        "vel_target_z",
        "acc_x",
        "acc_y",
        "acc_z",
    ]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=coords)
    df[coords + ["time"]] = df[coords + ["time"]].apply(pd.to_numeric, errors="coerce")

    # === Continue to animation ===
    t = df["time"].values
    t_actual_moth = t - start_send_time
    pursuer_pos = df[["pos_x", "pos_y", "pos_z"]].values
    evader_pos = df[["pos_target_x", "pos_target_y", "pos_target_z"]].values
    accel_cmd = df[["accel_command_x", "accel_command_y", "accel_command_z"]].values

    vel_cmd = df[["vel_x", "vel_y", "vel_z"]].values
    vel_tgt = df[["vel_target_x", "vel_target_y", "vel_target_z"]].values
    phi = -df["att_phi"].values
    theta = -df["att_theta"].values
    psi = df["att_psi"].values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("CTBR-NN REAL")

    all_xyz = np.vstack([pursuer_pos, evader_pos])
    pad = 0.5
    ax.set_xlim(all_xyz[:, 0].min() - pad, all_xyz[:, 0].max() + pad)
    ax.set_ylim(all_xyz[:, 1].min() - pad, all_xyz[:, 1].max() + pad)
    ax.set_zlim(all_xyz[:, 2].min() - pad, all_xyz[:, 2].max() + pad)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    floor = np.array(
        [
            [all_xyz[:, 0].min() - pad, all_xyz[:, 1].min() - pad, 0],
            [all_xyz[:, 0].min() - pad, all_xyz[:, 1].max() + pad, 0],
            [all_xyz[:, 0].max() + pad, all_xyz[:, 1].max() + pad, 0],
            [all_xyz[:, 0].max() + pad, all_xyz[:, 1].min() - pad, 0],
        ]
    )
    ax.add_collection3d(Poly3DCollection([floor], color="gray", alpha=0.2))
    (pursuer_line,) = ax.plot([], [], [], lw=2, color="red", label="Drone")
    (evader_line,) = ax.plot([], [], [], lw=2, color="blue", label="Target")
    (pursuer_marker,) = ax.plot([], [], [], "o", color="red")
    (evader_marker,) = ax.plot([], [], [], "o", color="blue")
    L, W = 0.4, 0.1
    body_shape = np.array([[L, -L / 2, -L / 2], [0, W / 2, -W / 2], [0, 0, 0]])
    drone_poly = Poly3DCollection([body_shape.T], color="green", alpha=0.8)
    ax.add_collection3d(drone_poly)
    accel_q = ax.quiver(0, 0, 0, 0, 0, 0, color="magenta")
    vel_q = ax.quiver(0, 0, 0, 0, 0, 0, color="cyan")
    tgt_q = ax.quiver(0, 0, 0, 0, 0, 0, color="yellow")
    intercepts = set(
        np.where(np.linalg.norm(pursuer_pos - evader_pos, axis=1) <= 0.15)[0]
    )
    intercept_scatter = ax.scatter(
        [], [], [], color="black", s=30, label="Interceptions"
    )

    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def rotation_matrix(ix):
        cp, sp = np.cos(psi[ix]), np.sin(psi[ix])
        ct, st = np.cos(theta[ix]), np.sin(theta[ix])
        cr, sr = np.cos(phi[ix]), np.sin(phi[ix])
        Rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
        Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    def init():
        for artist in [pursuer_line, evader_line, pursuer_marker, evader_marker]:
            artist.set_data([], [])
            artist.set_3d_properties([])
        drone_poly.set_verts([body_shape.T])
        intercept_scatter._offsets3d = ([], [], [])
        return pursuer_line, evader_line, intercept_scatter

    def update(ix):
        nonlocal accel_q, vel_q, tgt_q
        p = pursuer_pos[ix]
        e = evader_pos[ix]
        R = rotation_matrix(ix)

        window_sec = 3.0
        min_ix = np.searchsorted(t, t[ix] - window_sec)
        trail_slice = slice(min_ix, ix + 1)

        pursuer_line.set_data(pursuer_pos[trail_slice, 0], pursuer_pos[trail_slice, 1])
        pursuer_line.set_3d_properties(pursuer_pos[trail_slice, 2])

        evader_line.set_data(evader_pos[trail_slice, 0], evader_pos[trail_slice, 1])
        evader_line.set_3d_properties(evader_pos[trail_slice, 2])

        pursuer_marker.set_data([p[0]], [p[1]])
        pursuer_marker.set_3d_properties([p[2]])

        evader_marker.set_data([e[0]], [e[1]])
        evader_marker.set_3d_properties([e[2]])

        drone_poly.set_verts([(R @ body_shape).T + p])

        recent_intercepts = [i for i in intercepts if min_ix <= i <= ix]
        if recent_intercepts:
            pts = pursuer_pos[recent_intercepts]
            intercept_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        else:
            intercept_scatter._offsets3d = ([], [], [])

        a = accel_cmd[ix]
        a_dir = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.zeros(3)
        a_world = R @ a_dir

        accel_q.remove()
        vel_q.remove()
        tgt_q.remove()

        accel_q = ax.quiver(*p, *a_world, length=1, normalize=True, color="magenta")
        vel_q = ax.quiver(*p, *vel_cmd[ix], length=1, normalize=True, color="cyan")
        tgt_q = ax.quiver(*e, *vel_tgt[ix], length=1, normalize=True, color="yellow")

        time_text.set_text(f"Time: {t[ix]:.2f}s")
        return (
            pursuer_line,
            evader_line,
            pursuer_marker,
            evader_marker,
            drone_poly,
            accel_q,
            vel_q,
            tgt_q,
            time_text,
            intercept_scatter,
        )

    frames = range(0, len(t), speed_factor)
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        interval=interval,
        blit=False,
        repeat=False,
    )
    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="Drone"),
        Line2D([0], [0], color="blue", lw=2, label="Target (Received)"),
        Line2D([0], [0], color="magenta", lw=2, label="Accel Cmd"),
        Line2D([0], [0], color="cyan", lw=2, label="Drone Velocity"),
        Line2D([0], [0], color="yellow", lw=2, label="Target Velocity"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    if save_file:
        writer = "pillow" if save_file.endswith(".gif") else "ffmpeg"
        anim.save(save_file, writer=writer, fps=60)
    plt.show()
    return anim


def plot_diagnostics(df):
    t = df["time"].values

    def safe_extract(components, prefix):
        try:
            return df[[f"{prefix}_{a}" for a in components]].values.T
        except KeyError:
            return None

    def plot_xyz_components(figtitle, data_dict, ylabel, style_dict=None):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, ax in enumerate(axs):
            for label, series in data_dict.items():
                if series is not None:
                    linestyle = style_dict.get(label, "-") if style_dict else "-"
                    ax.plot(t, series[i], label=label, linestyle=linestyle)
            ax.set_ylabel(f"{ylabel} {['X', 'Y', 'Z'][i]}")
            ax.grid(True)
            ax.legend()
        axs[-1].set_xlabel("Time [s]")
        fig.suptitle(figtitle)

    # === Velocity comparisons ===
    plot_xyz_components(
        "Drone Velocity: Raw vs Filtered",
        {
            "Raw": safe_extract("xyz", "raw_pu_vel"),
            "Filtered": safe_extract("xyz", "filt_pu_vel"),
        },
        "Vel [m/s]",
        {"Raw": ":", "Filtered": "-"},
    )

    plot_xyz_components(
        "Target Velocity: Raw vs Filtered",
        {
            "Raw": safe_extract("xyz", "raw_ev_vel"),
            "Filtered": safe_extract("xyz", "filt_ev_vel"),
        },
        "Vel [m/s]",
        {"Raw": ":", "Filtered": "-"},
    )

    plot_xyz_components(
        "Relative Velocity (r_dot): Raw vs Filtered",
        {
            "Raw": safe_extract("xyz", "raw_r_dot"),
            "Filtered": safe_extract("xyz", "filt_r_dot"),
        },
        "ṙ [m/s]",
        {"Raw": ":", "Filtered": "-"},
    )

    # === Acceleration comparisons ===
    plot_xyz_components(
        "Acceleration: Commanded vs Measured",
        {
            "Commanded": safe_extract("xyz", "accel_command"),
            "Measured": safe_extract("xyz", "acc"),
        },
        "Acc [m/s²]",
        {"Commanded": "--", "Measured": "-"},
    )

    # === CTBR Neural Net Outputs ===
    ctbr_fields = ["nn_p", "nn_q", "nn_r", "nn_thrust"]
    if all(f in df.columns for f in ctbr_fields):
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        for i, field in enumerate(ctbr_fields):
            axs[i].plot(t, df[field], label=field, color="C" + str(i))
            axs[i].set_ylabel(field)
            axs[i].grid(True)
            axs[i].legend()
        axs[-1].set_xlabel("Time [s]")
        fig.suptitle("CTBR Neural Network Outputs")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        print("CTBR fields not present in this log file.")

        # === Attitude: phi, theta, psi ===
    if all(k in df.columns for k in ["att_phi", "att_theta", "att_psi"]):
        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        att_labels = ["att_phi", "att_theta", "att_psi"]
        att_names = ["Roll (φ)", "Pitch (θ)", "Yaw (ψ)"]
        for i, (col, label) in enumerate(zip(att_labels, att_names)):
            axs[i].plot(t, df[col], label=label)
            axs[i].set_ylabel(f"{label} [rad]")
            axs[i].grid(True)
            axs[i].legend()
        axs[-1].set_xlabel("Time [s]")
        fig.suptitle("Drone Attitude (Euler Angles)")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        print("Attitude columns not found in log.")

        # === Angular Rates: Measured vs NN Commanded ===
    rate_cols = ["rate_p", "rate_q", "rate_r"]
    cmd_cols = ["nn_p", "nn_q", "nn_r"]
    if all(k in df.columns for k in rate_cols + cmd_cols):
        fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for i, (meas_col, cmd_col) in enumerate(zip(rate_cols, cmd_cols)):
            axs[i].plot(t, df[meas_col], label="Measured", linestyle="-")
            axs[i].plot(t, df[cmd_col], label="NN Commanded", linestyle="--")
            axs[i].set_ylabel(["p", "q", "r"][i] + " [rad/s]")
            axs[i].grid(True)
            axs[i].legend()
        axs[-1].set_xlabel("Time [s]")
        fig.suptitle("Angular Rates: Measured vs NN Commanded")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        print("Angular rate or NN command columns not found.")


# === USAGE ===
pn_csv_path = "/home/merlijn/Desktop/log_files/20250701-120104.csv"

df = pd.read_csv(pn_csv_path)
# plot_diagnostics(df)

anim = animate_pn_log(
    pn_csv_path=pn_csv_path,
    start_send_time=448.586334,
    pos_offset=np.array([0.0, 0.0, 2.0]),
    scale_factor=1.0,
    speed_factor=60,
    interval=1,
    save_file=None,
)
