import numpy as np


def make_fov_planes_with_depth(C, f, r, u, hfov, vfov, near, far):
    """
    C: (3,) camera center
    f,r,u: (3,) unit forward/right/up in world (NED-like) coords (must be orthonormal)
    hfov,vfov: radians (full angles)
    near,far: positive distances along +f
    Returns list of 6 dicts {n,p0} with inward normals.
    Inside test convention: point x is inside if n · (x - p0) >= 0.
    """
    f = f / np.linalg.norm(f)
    r = r / np.linalg.norm(r)
    u = u / np.linalg.norm(u)

    # side planes pass through camera center, normals point inward
    n_left = np.cos(hfov / 2) * r + np.sin(hfov / 2) * f
    n_right = -np.cos(hfov / 2) * r + np.sin(hfov / 2) * f
    n_up = np.cos(vfov / 2) * u + np.sin(vfov / 2) * f
    n_down = -np.cos(vfov / 2) * u + np.sin(vfov / 2) * f

    planes = [
        {"n": n_left.tolist(), "p0": C.tolist()},
        {"n": n_right.tolist(), "p0": C.tolist()},
        {"n": n_up.tolist(), "p0": C.tolist()},
        {"n": n_down.tolist(), "p0": C.tolist()},
        {"n": f.tolist(), "p0": (C + f * near).tolist()},  # near
        {"n": (-f).tolist(), "p0": (C + f * far).tolist()},  # far
    ]
    return planes


def add_horizontal_floor(planes, C, floor_z):
    """
    Add a horizontal floor at z = floor_z (with +z pointing downward).
    Keeps points ABOVE the floor (z <= floor_z).
    Inside rule: n · (x - p0) >= 0, with n = [0,0,-1] (up).
    """
    n_floor = np.array([0.0, 0.0, -1.0])  # inward normal (up)
    p0_floor = C + np.array([0.0, 0.0, floor_z])  # a point on the floor
    planes.append({"n": n_floor.tolist(), "p0": p0_floor.tolist()})


def add_horizontal_roof(planes, C, roof_z):
    """
    Add a horizontal roof at z = roof_z (with +z pointing downward).
    Keeps points BELOW the roof (z >= roof_z).
    Inside rule: n · (x - p0) >= 0, with n = [0,0,1] (down).
    """
    n_roof = np.array([0.0, 0.0, 1.0])  # inward normal (down)
    p0_roof = C + np.array([0.0, 0.0, roof_z])  # a point on the roof
    planes.append({"n": n_roof.tolist(), "p0": p0_roof.tolist()})


if __name__ == "__main__":

    camera_center = np.array([0.0, 0.0, 0.0])

    # Camera pitched 15° down from horizon (NED-like: +z is down)
    deg = np.pi / 180.0
    tilt_angle = 30
    f = np.array(
        [np.cos(tilt_angle * deg), 0.0, np.sin(tilt_angle * deg)]
    )  # forward (tilted downward)
    r = np.array([0.0, 1.0, 0.0])  # right
    u = np.array([np.sin(tilt_angle * deg), 0.0, -np.cos(tilt_angle * deg)])  # up

    # FOV and depth
    hfov = 87 * np.pi / 180.0
    vfov = 58 * np.pi / 180.0
    near = 0.55
    far = 4.0

    planes = make_fov_planes_with_depth(camera_center, f, r, u, hfov, vfov, near, far)

    # --- Extra environmental planes ---
    floor_z = 2.0
    roof_z = -0.1

    add_horizontal_floor(planes, camera_center, floor_z=floor_z)
    add_horizontal_roof(planes, camera_center, roof_z=roof_z)

    x_back = 2.25  # 1 m behind the camera along -N
    back_wall_plane = {
        "n": [-1.0, 0.0, 0.0],
        "p0": [x_back, float(camera_center[1]), float(camera_center[2])],
    }

    planes.append(back_wall_plane)

    # Print
    print(planes)
