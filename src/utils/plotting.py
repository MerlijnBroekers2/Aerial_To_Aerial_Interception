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


def plot_motor_omega(sim_history, save_path=None):
    """
    Plot the motor angular velocities (omega) over time for the pursuer.

    Parameters
    ----------
    sim_history : list of dicts
        The simulation history as stored in Simulation.history.
    save_path : str or None
        If provided, save the figure to this path.
    """
    times = [entry["time"] for entry in sim_history]
    omega = np.array(
        [entry["p_state"]["omega"] for entry in sim_history]
    )  # shape: (N, 4)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(4):
        ax.plot(times, omega[:, i], label=f"Motor {i+1}")

    ax.set_title("Motor Angular Velocities Over Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Omega [rad/s]")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


from mpl_toolkits.mplot3d.art3d import Line3DCollection  # you already import this later


def _box_edges_from_limits(xmin, xmax, ymin, ymax, zmin, zmax):
    """Return the 12 axis-aligned box edges as line segments (12,2,3)."""
    return _box_edge_segments(xmin, xmax, ymin, ymax, zmin, zmax)


def _intersect_segment_with_plane(p0, p1, plane_n, plane_p0):
    """
    Intersect segment p(t)=p0 + t*(p1-p0), t in [0,1], with plane n·(x-p0)=0.
    Returns (hit, point).
    """
    v = p1 - p0
    denom = float(np.dot(plane_n, v))
    if abs(denom) < 1e-12:
        return False, None  # parallel or coincident (ignore)
    t = float(np.dot(plane_n, (plane_p0 - p0))) / denom
    if 0.0 <= t <= 1.0:
        return True, p0 + t * v
    return False, None


def _plane_polygon_in_box(plane, limits):
    """
    Clip a plane to an axis-aligned box. Returns Nx3 polygon vertices (CCW-ish).
    plane: {"n":[...], "p0":[...]} inward normal not required here
    limits: (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    xmin, xmax, ymin, ymax, zmin, zmax = limits
    edges = _box_edges_from_limits(xmin, xmax, ymin, ymax, zmin, zmax)  # (12,2,3)
    n = np.asarray(plane["n"], float)
    p0 = np.asarray(plane["p0"], float)
    n /= np.linalg.norm(n) + 1e-12

    pts = []
    for seg in edges:
        hit, pt = _intersect_segment_with_plane(seg[0], seg[1], n, p0)
        if hit:
            pts.append(pt)
    if len(pts) < 3:
        return None  # plane misses the box

    P = np.vstack(pts)

    # Order points around their centroid using a 2D basis on the plane
    c = P.mean(axis=0)
    # build orthonormal basis (u,v) spanning the plane
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, n) * n
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    # project to 2D
    U = (P - c) @ np.vstack([u, v]).T  # (N,2)
    ang = np.arctan2(U[:, 1], U[:, 0])
    order = np.argsort(ang)
    return P[order]


def _frustum_polygons(corners):
    """
    Build the 6 plane quads from the 8 frustum corners.
    Returns dict: name -> (ordered 4x3 vertices)
    Names: 'left','right','up','down','near','far'
    """
    # Corner names: nlu, nld, nru, nrd, flu, fld, fru, frd
    C = corners
    polys = {
        # keep vertex order roughly CCW as seen from inside
        "left": np.array([C["nlu"], C["nld"], C["fld"], C["flu"]]),
        "right": np.array([C["nru"], C["nrd"], C["frd"], C["fru"]]),
        "up": np.array([C["nlu"], C["nru"], C["fru"], C["flu"]]),
        "down": np.array([C["nld"], C["nrd"], C["frd"], C["fld"]]),
        "near": np.array([C["nlu"], C["nru"], C["nrd"], C["nld"]]),
        "far": np.array([C["flu"], C["fru"], C["frd"], C["fld"]]),
    }
    return polys


def _plane_signed_distance(x, plane):
    """Return signed distance n·(x - p0). Inside if >= 0 (by our convention)."""
    n = np.array(plane["n"], dtype=float)
    p0 = np.array(plane["p0"], dtype=float)
    return float(np.dot(n, x - p0))


def _parse_env_bounds(sim_result, env_bounds, xyz_all):
    # Priority: explicit arg > config in sim_result > fallback to data extents
    if env_bounds is None:
        env_bounds = (
            sim_result.get("config", {}).get("BOUNDARIES", {}).get("ENV_BOUNDS", None)
        )

    if isinstance(env_bounds, dict):
        xmin, xmax = env_bounds["x"]
        ymin, ymax = env_bounds["y"]
        zmin, zmax = env_bounds["z"]
    elif isinstance(env_bounds, tuple) or isinstance(env_bounds, list):
        # Expect ((xmin,ymin,zmin), (xmax,ymax,zmax))
        mins, maxs = env_bounds
        xmin, ymin, zmin = mins
        xmax, ymax, zmax = maxs
    else:
        # Fallback: use data extents (not ideal, but keeps it robust)
        pad = 0.0
        xmin, xmax = xyz_all[:, 0].min() - pad, xyz_all[:, 0].max() + pad
        ymin, ymax = xyz_all[:, 1].min() - pad, xyz_all[:, 1].max() + pad
        zmin, zmax = xyz_all[:, 2].min() - pad, xyz_all[:, 2].max() + pad

    return float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)


def _box_edge_segments(xmin, xmax, ymin, ymax, zmin, zmax):
    # 12 edges of the axis-aligned box as segments: shape (12, 2, 3)
    xs = [xmin, xmax]
    ys = [ymin, ymax]
    zs = [zmin, zmax]
    c = np.array(
        [[x, y, z] for x in xs for y in ys for z in zs], dtype=float
    )  # 8 corners
    # Corner indices in (x,y,z) binary order: 0..7
    # Edges connect corners that differ in exactly one coordinate
    edges_idx = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),  # x-edges at y,z planes
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),  # y-edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # z-edges
    ]
    return np.stack([np.stack([c[i], c[j]], axis=0) for i, j in edges_idx], axis=0)


def _plane_intersection(pA, pB, pC):
    """
    Intersect 3 planes given as dicts:
      p['n'] -> normal (3,), p['p0'] -> any point on plane (3,)
    Returns x solving [nA; nB; nC] x = [nA·p0A; nB·p0B; nC·p0C]
    """
    nA, nB, nC = (
        np.array(pA["n"], float),
        np.array(pB["n"], float),
        np.array(pC["n"], float),
    )
    b = np.array(
        [np.dot(nA, pA["p0"]), np.dot(nB, pB["p0"]), np.dot(nC, pC["p0"])], float
    )
    M = np.vstack([nA, nB, nC])
    return np.linalg.solve(M, b)


def _corners_from_planes(planes):
    """
    planes: list of six dicts in order [left, right, up, down, near, far]
    Returns dict of 8 corners and a list of 12 edge segments.
    """
    L, R, U, D, Np, Fp = planes

    # 8 corners = {near,far} × {left,right} × {up,down}
    corners = {
        "nlu": _plane_intersection(Np, L, U),
        "nld": _plane_intersection(Np, L, D),
        "nru": _plane_intersection(Np, R, U),
        "nrd": _plane_intersection(Np, R, D),
        "flu": _plane_intersection(Fp, L, U),
        "fld": _plane_intersection(Fp, L, D),
        "fru": _plane_intersection(Fp, R, U),
        "frd": _plane_intersection(Fp, R, D),
    }

    # Build 12 edges (each as 2-point segment)
    seg = lambda a, b: np.stack([corners[a], corners[b]], axis=0)
    edge_segments = [
        # near rectangle
        seg("nlu", "nru"),
        seg("nru", "nrd"),
        seg("nrd", "nld"),
        seg("nld", "nlu"),
        # far rectangle
        seg("flu", "fru"),
        seg("fru", "frd"),
        seg("frd", "fld"),
        seg("fld", "flu"),
        # 4 “rays” connecting near↔far
        seg("nlu", "flu"),
        seg("nru", "fru"),
        seg("nrd", "frd"),
        seg("nld", "fld"),
    ]
    return corners, edge_segments


def _frustum_limits(corners, xyz_all, pad=0.25):
    pts = np.vstack([xyz_all, *corners.values()]) if corners else xyz_all
    mins = pts.min(axis=0) - pad
    maxs = pts.max(axis=0) + pad
    return mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]


# --- add this near your imports ---
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D


def _circle_xy(radius=0.05, z=0.0, n=40):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=True)
    x = radius * np.cos(ang)
    y = radius * np.sin(ang)
    z = np.full_like(x, z)
    return np.vstack([x, y, z]).T  # (n,3)


def _transform_pts(pts, R, t):
    return (R @ pts.T).T + t


class QuadModel:
    """
    Minimal quadcopter: two arms (x- and y-axis), four rotor disks, small hub.
    All geometry defined in the BODY frame (x forward, y right, z up), then
    transformed by R, t each frame.
    """

    def __init__(
        self,
        ax,
        arm=0.10,
        hub_r=0.02,
        prop_r=0.12 / 5 * 2,
        arm_w=0.02 / 5 * 2,
        color_arm="dimgray",
        color_prop="lightgray",
        color_hub="red",
        prop_segments=36,
    ):
        self.ax = ax
        self.arm = float(arm)
        self.hub_r = float(hub_r)
        self.prop_r = float(prop_r)
        self.arm_w = float(arm_w)

        # ---- define BODY-frame geometry ----
        # Arms as thin rectangles (4 verts) extruded as very thin "ribbons"
        a = arm
        w = arm_w
        # X-arm rectangle in XY plane
        self._xarm_body = np.array(
            [[-a, -w / 2, 0], [a, -w / 2, 0], [a, w / 2, 0], [-a, w / 2, 0]]
        )
        # Y-arm rectangle
        self._yarm_body = np.array(
            [[-w / 2, -a, 0], [-w / 2, a, 0], [w / 2, a, 0], [w / 2, -a, 0]]
        )

        # Hub as a flat hexagon
        th = np.linspace(0, 2 * np.pi, 7)
        self._hub_body = np.vstack(
            [self.hub_r * np.cos(th), self.hub_r * np.sin(th), np.zeros_like(th)]
        ).T

        # Rotor disks (circles) centered at arm tips (z slightly above arms)
        z_prop = 0.0 + 0.0  # keep flat; raise slightly if you want layering
        circle = _circle_xy(self.prop_r, z=z_prop, n=prop_segments)
        self._mot_centers = np.array(
            [
                [a, 0, 0],
                [-a, 0, 0],
                [0, a, 0],
                [0, -a, 0],
            ]
        )
        self._props_body = [circle + c for c in self._mot_centers]

        # ---- artists ----
        self.poly_xarm = Poly3DCollection(
            [self._xarm_body],
            facecolor=color_arm,
            edgecolor="k",
            linewidths=0.4,
            alpha=0.95,
        )
        self.poly_yarm = Poly3DCollection(
            [self._yarm_body],
            facecolor=color_arm,
            edgecolor="k",
            linewidths=0.4,
            alpha=0.95,
        )
        self.poly_hub = Poly3DCollection(
            [self._hub_body],
            facecolor=color_hub,
            edgecolor="k",
            linewidths=0.4,
            alpha=0.95,
        )
        self.poly_props = [
            Poly3DCollection(
                [p], facecolor=color_prop, edgecolor="k", linewidths=0.3, alpha=0.85
            )
            for p in self._props_body
        ]

        for artist in (self.poly_xarm, self.poly_yarm, self.poly_hub, *self.poly_props):
            ax.add_collection3d(artist)

    def set_pose(self, R, t):
        # R: (3,3) body->world rotation; t: (3,) world position
        def set_poly(poly, verts):
            poly.set_verts([verts])

        set_poly(self.poly_xarm, _transform_pts(self._xarm_body, R, t))
        set_poly(self.poly_yarm, _transform_pts(self._yarm_body, R, t))
        set_poly(self.poly_hub, _transform_pts(self._hub_body, R, t))
        for poly, pb in zip(self.poly_props, self._props_body):
            set_poly(poly, _transform_pts(pb, R, t))


# --------------------------------------------------------------------------
def animate_pursuit_evasion(
    sim_result: dict,
    *,
    pos_offset: np.ndarray | None = None,
    scale_factor: float = 1.0,
    speed_factor: int = 1,
    interval: int = 50,
    save_file: str | None = None,
    env_planes: (
        list[dict] | None
    ) = None,  # inward normals; draws only the enclosed outline
    env_bounds: dict | tuple | None = None,  # legacy box fallback
    env_bounds_margin: float = 1.5,
):
    """
    Visualise a single pursuit–evasion run.

    Out-of-bounds styling for BOTH agents:
      - hollow marker (edge-only) + dotted, faint trail.
    In-bounds styling:
      - filled marker + solid trail.
    """

    # ------------------------------ data ---------------------------------
    H = sim_result["history"]
    N = len(H)

    t = np.array([h["time"] for h in H])
    pursuer_pos = np.stack([h["p_state"]["noisy_position"] for h in H])
    evader_pos = np.stack([h["e_state"]["filtered_position"] for h in H])
    attitude = np.stack([h["p_state"]["attitude"] for h in H])
    v_pursuer = np.stack([h["p_state"]["velocity"] for h in H])
    v_evader = np.stack([h["e_state"]["filtered_velocity"] for h in H])
    a_cmd = np.stack([h["p_state"]["acc_command"] for h in H])

    if pos_offset is not None:
        pursuer_pos += pos_offset
        evader_pos += pos_offset
    pursuer_pos *= scale_factor
    evader_pos *= scale_factor

    # ------------------- deduped interceptions for markers ----------------
    min_gap = 0.25
    if sim_result.get("interceptions"):
        hits = sorted(sim_result["interceptions"], key=lambda h: float(h["time"]))
        kept, last_t = [], -np.inf
        for h in hits:
            t_h = float(h["time"])
            if t_h - last_t >= min_gap:
                kept.append(h)
                last_t = t_h
        if kept:
            hit_times = np.array([float(h["time"]) for h in kept], dtype=float)
            hit_points = np.stack([h["pursuer_pos"] for h in kept], axis=0)
        else:
            hit_times = np.empty(0, dtype=float)
            hit_points = np.empty((0, 3), dtype=float)
    else:
        hit_times = np.empty(0, dtype=float)
        hit_points = np.empty((0, 3), dtype=float)

    # ------------------------------ figure --------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Simulation run RL_CTBR")

    xyz_all = np.vstack([pursuer_pos, evader_pos])

    # ---------- helper: build enclosed outline from planes (wireframe) ----
    def _polyhedron_wireframe(planes: list[dict], tol=1e-7):
        import itertools

        n_list = [np.asarray(p["n"], float) for p in planes]
        p0_list = [np.asarray(p["p0"], float) for p in planes]
        n_list = [n / (np.linalg.norm(n) + 1e-12) for n in n_list]
        P = len(n_list)
        verts = []
        for a, b, c in itertools.combinations(range(P), 3):
            N3 = np.vstack([n_list[a], n_list[b], n_list[c]])
            if abs(np.linalg.det(N3)) < 1e-10:
                continue
            rhs = np.array(
                [
                    np.dot(n_list[a], p0_list[a]),
                    np.dot(n_list[b], p0_list[b]),
                    np.dot(n_list[c], p0_list[c]),
                ],
                float,
            )
            x = np.linalg.solve(N3, rhs)
            if all(np.dot(n_list[i], x - p0_list[i]) >= -tol for i in range(P)):
                verts.append(x)
        if not verts:
            return None, None
        V = np.array(verts, float)
        key = np.round(V / (10 * tol))
        _, unique_idx = np.unique(key, axis=0, return_index=True)
        V = V[sorted(unique_idx)]

        edges = []
        plane_hit_eps = 5e-4
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                vi, vj = V[i], V[j]
                on_i = [
                    k
                    for k in range(P)
                    if abs(np.dot(n_list[k], vi - p0_list[k])) <= plane_hit_eps
                ]
                on_j = [
                    k
                    for k in range(P)
                    if abs(np.dot(n_list[k], vj - p0_list[k])) <= plane_hit_eps
                ]
                if len(set(on_i).intersection(on_j)) >= 2:
                    m = 0.5 * (vi + vj)
                    if all(
                        np.dot(n_list[k], m - p0_list[k]) >= -1e-6 for k in range(P)
                    ):
                        edges.append(np.stack([vi, vj], axis=0))
        if not edges:
            return V, None
        return V, np.stack(edges, axis=0)

    outline_edges = None
    vertices = None

    if env_planes and len(env_planes) >= 4:
        try:
            vertices, outline_edges = _polyhedron_wireframe(env_planes)
        except Exception:
            vertices, outline_edges = None, None

    if outline_edges is not None and vertices is not None and len(vertices) >= 4:
        pts = np.vstack([xyz_all, vertices])
        pad = 0.25
        mins = pts.min(axis=0) - pad
        maxs = pts.max(axis=0) + pad
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(maxs[2], mins[2])

        # dotted wireframe of in-bounds volume
        bounds_lines = Line3DCollection(
            outline_edges, colors="k", linestyles=":", linewidths=1.4, alpha=0.9
        )
        ax.add_collection3d(bounds_lines)
        limits = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
    else:
        # fallback: legacy box outline
        xmin, xmax, ymin, ymax, zmin, zmax = _parse_env_bounds(
            sim_result, env_bounds, xyz_all
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        edge_segments = _box_edge_segments(
            xmin + env_bounds_margin,
            xmax - env_bounds_margin,
            ymin + env_bounds_margin,
            ymax - env_bounds_margin,
            zmin + env_bounds_margin,
            zmax - env_bounds_margin,
        )
        bounds_lines = Line3DCollection(
            edge_segments, colors="k", linestyles=":", linewidths=1.2, alpha=0.8
        )
        ax.add_collection3d(bounds_lines)
        limits = (xmin, xmax, ymin, ymax, zmin, zmax)

    if env_planes and len(env_planes) >= 4:
        n_arr = np.array([p["n"] for p in env_planes], dtype=float)
        p0_arr = np.array([p["p0"] for p in env_planes], dtype=float)
        n_arr = n_arr / (np.linalg.norm(n_arr, axis=1, keepdims=True) + 1e-12)

        def _inside_soft(x):
            # soft: require margin clearance on all planes
            s = np.einsum("ij,ij->i", (x - p0_arr), n_arr)  # (P,)
            return np.all(s >= env_bounds_margin)

        def _inside_hard(x):
            # hard: only require x to be inside half-spaces (no margin)
            s = np.einsum("ij,ij->i", (x - p0_arr), n_arr)  # (P,)
            return np.all(s >= 0.0)

    else:
        xmin, xmax, ymin, ymax, zmin, zmax = limits

        def _inside_soft(x):
            return (
                (x[0] >= xmin + env_bounds_margin)
                and (x[0] <= xmax - env_bounds_margin)
                and (x[1] >= ymin + env_bounds_margin)
                and (x[1] <= ymax - env_bounds_margin)
                and (x[2] >= zmin + env_bounds_margin)
                and (x[2] <= zmax - env_bounds_margin)
            )

        def _inside_hard(x):
            return (
                (x[0] >= xmin)
                and (x[0] <= xmax)
                and (x[1] >= ymin)
                and (x[1] <= ymax)
                and (x[2] >= zmin)
                and (x[2] <= zmax)
            )

    # ------------------------------ legend/labels --------------------------
    ax.legend(
        handles=[
            Line2D([0], [0], color="red", lw=2, label="Pursuer (in-bounds)"),
            Line2D(
                [0],
                [0],
                color="red",
                lw=2,
                ls=":",
                alpha=0.35,
                label="Pursuer (out-of-bounds)",
            ),
            Line2D([0], [0], color="blue", lw=2, label="Evader (in-bounds)"),
            Line2D(
                [0],
                [0],
                color="blue",
                lw=2,
                ls=":",
                alpha=0.35,
                label="Evader (out-of-bounds)",
            ),
            Line2D([0], [0], color="magenta", lw=2, label="Accel cmd"),
            Line2D([0], [0], color="cyan", lw=2, label="Pursuer vel"),
            Line2D([0], [0], color="yellow", lw=2, label="Evader vel"),
        ],
        loc="upper right",
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # --------------------------- actors & gizmos ---------------------------
    # PURSUER trails (split)
    (p_line_in,) = ax.plot([], [], [], lw=2, color="red")
    (p_line_out,) = ax.plot([], [], [], lw=2, color="red", ls=":", alpha=0.35)

    # EVADER trails (split)
    (e_line_in,) = ax.plot([], [], [], lw=2, color="blue")
    (e_line_out,) = ax.plot([], [], [], lw=2, color="blue", ls=":", alpha=0.35)

    # markers (we flip face/edge each frame)
    (p_mark,) = ax.plot(
        [],
        [],
        [],
        marker="o",
        linestyle="None",
        markerfacecolor="red",
        markeredgecolor="red",
    )
    (e_mark,) = ax.plot(
        [],
        [],
        [],
        marker="o",
        linestyle="None",
        markerfacecolor="blue",
        markeredgecolor="blue",
    )

    # tiny triangular “airframe” on pursuer
    quad = QuadModel(ax)

    # arrows
    accel_line = Line3D([], [], [], color="magenta")
    vel_line = Line3D([], [], [], color="cyan")
    tgt_line = Line3D([], [], [], color="yellow")
    ax.add_line(accel_line)
    ax.add_line(vel_line)
    ax.add_line(tgt_line)

    # interceptions
    hit_scatter = ax.scatter([], [], [], marker="x", s=80, c="black")

    time_txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # ------------------------------ helpers --------------------------------
    def _rot(phi, theta, psi):
        c, s = np.cos, np.sin
        Rz = np.array([[c(psi), -s(psi), 0], [s(psi), c(psi), 0], [0, 0, 1]])
        Ry = np.array([[c(theta), 0, s(theta)], [0, 1, 0], [-s(theta), 0, c(theta)]])
        Rx = np.array([[1, 0, 0], [0, c(phi), -s(phi)], [0, s(phi), c(phi)]])
        return Rz @ Ry @ Rx

    def _masked_polyline(P, mask):
        """
        P: (M,3) points; mask: (M,) bool for 'inside'.
        Returns x,y,z arrays with NaN breaks for segments that don't match mask.
        """
        if len(P) < 2:
            return [], [], []
        xs, ys, zs = [], [], []

        def _append_nan():
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)

        prev_kept = False
        for k in range(len(P) - 1):
            keep = bool(mask[k] and mask[k + 1])  # both endpoints in same region
            if keep:
                if not prev_kept and len(xs) > 0:
                    _append_nan()
                xs.extend([P[k, 0], P[k + 1, 0]])
                ys.extend([P[k, 1], P[k + 1, 1]])
                zs.extend([P[k, 2], P[k + 1, 2]])
            prev_kept = keep
        return xs, ys, zs

    trail_window = 3.0  # seconds of trail

    def init():
        for ln in (
            p_line_in,
            p_line_out,
            e_line_in,
            e_line_out,
            accel_line,
            vel_line,
            tgt_line,
        ):
            ln.set_data_3d([], [], [])
        for mk in (p_mark, e_mark):
            mk.set_data_3d([], [], [])
        hit_scatter._offsets3d = ([], [], [])
        time_txt.set_text("")
        return (
            p_line_in,
            p_line_out,
            e_line_in,
            e_line_out,
            p_mark,
            e_mark,
            quad.poly_xarm,
            quad.poly_yarm,
            quad.poly_hub,
            *quad.poly_props,
            accel_line,
            vel_line,
            tgt_line,
            hit_scatter,
            time_txt,
        )

    def update(frame_idx):
        i = min(frame_idx, N - 1)

        # orientation & airframe
        phi, theta, psi = attitude[i]
        R = _rot(phi, theta, psi)
        quad.set_pose(R, pursuer_pos[i])

        # trails (sliding window)
        start = np.searchsorted(t, t[i] - trail_window)
        Pp = pursuer_pos[start : i + 1]
        Pe = evader_pos[start : i + 1]

        # inside masks over window
        p_inside_mask = np.array([_inside_hard(p) for p in Pp], dtype=bool)
        e_inside_mask = np.array([_inside_soft(p) for p in Pe], dtype=bool)

        # PURSUER trail split
        xpin, ypin, zpin = _masked_polyline(Pp, p_inside_mask)
        xpout, ypout, zpout = _masked_polyline(Pp, ~p_inside_mask)
        p_line_in.set_data_3d(xpin, ypin, zpin)
        p_line_out.set_data_3d(xpout, ypout, zpout)

        # EVADER trail split
        xein, yein, zein = _masked_polyline(Pe, e_inside_mask)
        xeout, yeout, zeout = _masked_polyline(Pe, ~e_inside_mask)
        e_line_in.set_data_3d(xein, yein, zein)
        e_line_out.set_data_3d(xeout, yeout, zeout)

        # markers (flip styles by in/out)
        p_in = bool(_inside_hard(pursuer_pos[i]))
        e_in = bool(_inside_soft(evader_pos[i]))

        p_mark.set_data_3d(
            [pursuer_pos[i, 0]], [pursuer_pos[i, 1]], [pursuer_pos[i, 2]]
        )
        if p_in:
            p_mark.set_markerfacecolor("red")
            p_mark.set_markeredgecolor("red")
            p_mark.set_markersize(6)
        else:
            p_mark.set_markerfacecolor("none")
            p_mark.set_markeredgecolor("red")
            p_mark.set_markersize(7)

        e_mark.set_data_3d([evader_pos[i, 0]], [evader_pos[i, 1]], [evader_pos[i, 2]])
        if e_in:
            e_mark.set_markerfacecolor("blue")
            e_mark.set_markeredgecolor("blue")
            e_mark.set_markersize(6)
        else:
            e_mark.set_markerfacecolor("none")
            e_mark.set_markeredgecolor("blue")
            e_mark.set_markersize(7)

        # arrows (body → world)
        a_dir = a_cmd[i] / 10
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

        # interceptions within window
        if hit_times.size:
            recent = (hit_times <= t[i]) & (hit_times >= t[i] - trail_window)
            if recent.any():
                pts = hit_points[recent]
                hit_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            else:
                hit_scatter._offsets3d = ([], [], [])

        time_txt.set_text(f"time : {t[i]:.2f} s")
        return (
            p_line_in,
            p_line_out,
            e_line_in,
            e_line_out,
            p_mark,
            e_mark,
            quad.poly_xarm,
            quad.poly_yarm,
            quad.poly_hub,
            *quad.poly_props,
            accel_line,
            vel_line,
            tgt_line,
            hit_scatter,
            time_txt,
        )

    # ----------------------------- animation -------------------------------
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

    if save_file:
        writer = (
            PillowWriter(fps=60) if save_file.endswith(".gif") else FFMpegWriter(fps=60)
        )
        anim.save(save_file, writer=writer)

    plt.show()
    return anim


def plot_all_accelerations(sim_history, save_path=None):
    times = [entry["time"] for entry in sim_history]
    commanded = np.array([entry["p_state"]["acc_command"] for entry in sim_history])
    filtered = np.array(
        [entry["p_state"]["acc_command_filtered"] for entry in sim_history]
    )
    measured = np.array([entry["p_state"]["acc_measured"] for entry in sim_history])

    axes = ["x", "y", "z"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for i in range(3):
        axs[i].plot(
            times, commanded[:, i], label=f"Commanded a{axes[i]}", linestyle="-"
        )
        axs[i].plot(times, filtered[:, i], label=f"Filtered a{axes[i]}", linestyle="-.")
        axs[i].plot(times, measured[:, i], label=f"Measured a{axes[i]}", linestyle="--")
        axs[i].set_ylabel(f"a{axes[i]} [m/s²]")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Commanded vs Filtered vs Measured Acceleration")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_interception_clusters(
    result,
    cluster_radius=0.15,
    ax=None,
    cmap_name="viridis",
    point_size=80,
    smooth_sigma=2.0,
):
    """
    Plot the evader trajectory with speed-based color and interception clusters.

    Parameters
    ----------
    result : dict
        One simulation result containing history and interceptions.
    cluster_radius : float
        DBSCAN eps parameter in meters.
    ax : Axes3D or None
        If None, a new 3D plot is created.
    cmap_name : str
        Colormap for trajectory (default: 'coolwarm').
    point_size : int
        Size of interception dots.
    smooth_sigma : float
        Standard deviation for Gaussian smoothing of speed.
    """
    evader_pos = np.array([h["e_state"]["true_position"] for h in result["history"]])
    evader_vel = np.array([h["e_state"]["velocity"] for h in result["history"]])
    speed = np.linalg.norm(evader_vel, axis=1)
    speed_smooth = gaussian_filter1d(speed, sigma=smooth_sigma)

    # Color normalization
    norm = mcolors.Normalize(vmin=np.min(speed_smooth), vmax=np.max(speed_smooth))
    cmap = plt.get_cmap(cmap_name)

    # Segment-wise trajectory for color mapping
    points = evader_pos
    segments = np.stack([points[:-1], points[1:]], axis=1)
    speeds_segment = speed_smooth[:-1]  # one per segment
    colors_segment = cmap(norm(speeds_segment))

    # 3D colored line segments
    line_collection = Line3DCollection(segments, colors=colors_segment, linewidth=2)

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.add_collection3d(line_collection)

    # Plot interception clusters
    interceptions = result["interceptions"]
    if interceptions:
        int_pos = np.array([i["evader_pos"] for i in interceptions])
        pursuer_pos = np.array(
            [h["p_state"]["true_position"] for h in result["history"]]
        )
        labels = DBSCAN(eps=cluster_radius, min_samples=1).fit(int_pos).labels_
        n_clusters = labels.max() + 1

        best_points = []
        for lab in range(n_clusters):
            idx = np.where(labels == lab)[0]
            hist_idx = [interceptions[j]["idx"] for j in idx]
            dists = np.linalg.norm(pursuer_pos[hist_idx] - evader_pos[hist_idx], axis=1)
            best_i = np.argmin(dists)
            best_points.append((hist_idx[best_i], dists[best_i]))

        best_dists = np.array([d for _, d in best_points])
        dist_norm = colors.Normalize(vmin=best_dists.min(), vmax=best_dists.max())
        dist_cmap = cm.get_cmap(cmap_name)

        for hist_idx, d in best_points:
            p = evader_pos[hist_idx]
            ax.scatter(
                p[0],
                p[1],
                p[2],
                s=point_size,
                marker="o",
                color=dist_cmap(dist_norm(d)),
                edgecolors="k",
                linewidth=0.4,
            )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        "Moth Evader Trajectory and Interception Clusters of RL-Motor Commanded Pursuer"
    )

    # Shrink main plot to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0 + 0.05, box.y0, box.width * 0.85, box.height * 0.9])

    # Add left-side colorbar for speed
    speed_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_ax_left = fig.add_axes(
        [0.12, 0.25, 0.02, 0.5]
    )  # [left, bottom, width, height]
    cbar_left = fig.colorbar(speed_mappable, cax=cbar_ax_left)
    cbar_left.set_label("Smoothed Evader Speed [m/s]")

    # Add right-side colorbar for interception distances
    if interceptions:
        dist_mappable = cm.ScalarMappable(norm=dist_norm, cmap=dist_cmap)
        cbar_ax_right = fig.add_axes([0.88, 0.25, 0.02, 0.5])  # right side
        cbar_right = fig.colorbar(dist_mappable, cax=cbar_ax_right)
        cbar_right.set_label("Best distance in cluster (m)")

    plt.show()
    return ax


def plot_attitude(sim_history, save_path=None):
    times = [entry["time"] for entry in sim_history]
    attitude = np.array([entry["p_state"]["attitude"] for entry in sim_history])
    labels = ["Roll (φ)", "Pitch (θ)", "Yaw (ψ)"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs[i].plot(times, attitude[:, i])
        axs[i].set_ylabel(f"{labels[i]} [rad]")
        axs[i].grid(True)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Pursuer Attitude Over Time")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_angular_rates(sim_history, save_path=None):
    times = [entry["time"] for entry in sim_history]

    commanded = np.array([entry["p_state"]["rates_command"] for entry in sim_history])
    measured = np.array([entry["p_state"]["rates"] for entry in sim_history])
    labels = ["Roll Rate (p)", "Pitch Rate (q)", "Yaw Rate (r)"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs[i].plot(times, commanded[:, i], label="Commanded", linestyle="-")
        axs[i].plot(times, measured[:, i], label="Measured", linestyle="--")
        axs[i].set_ylabel(f"{labels[i]} [rad/s]")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Angular Rates: Commanded vs Measured")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_thrust(sim_history, save_path=None):
    times = [entry["time"] for entry in sim_history]
    commanded = np.array([entry["p_state"]["T_command"] for entry in sim_history])
    measured = np.array([entry["p_state"]["T_force"] for entry in sim_history])

    plt.figure(figsize=(10, 4))
    plt.plot(times, commanded, label="Commanded Thrust", linestyle="-")
    plt.plot(times, measured, label="Measured Thrust", linestyle="--")
    plt.title("Normalized Thrust: Commanded vs Measured")
    plt.xlabel("Time [s]")
    plt.ylabel("Thrust")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_range_and_los(sim_history, save_path=None):
    times = [entry["time"] for entry in sim_history]
    range_vals = [entry["guidance_state"]["R"] for entry in sim_history]
    los_rates = [
        np.linalg.norm(entry["guidance_state"]["phi_dot"]) for entry in sim_history
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(times, range_vals, label="Range", color="tab:blue")
    ax1.set_ylabel("Range [m]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(times, los_rates, label="LOS Rate", color="tab:red")
    ax2.set_ylabel("LOS Rate [rad/s]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Range and Line-of-Sight Rate Over Time")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
