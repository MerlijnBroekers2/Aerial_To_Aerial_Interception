"""
NOTE THE FOLLOWING IS TAKEN FROM https://arxiv.org/abs/2504.21586

Edits made to support:
- Parsing logs from paparazzi instead of BetaFlight
- Different filtering of logged data b/c different logging frequency and trustworthiness

Additional features added:
- System identification for first order repsonses to rate commands
"""

import csv, numpy as np, scipy.signal as sig
import json
from matplotlib import pyplot as plt
import scipy as sp
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import t as t_dist
from scipy.optimize import least_squares

PRINT_CI = True  # toggle wether to receive infornmation about the CI of the different parameters


def ols_with_ci(X, y, alpha=0.05):
    """
    X: (n, p) design matrix (no intercept unless you include a column of 1s)
    y: (n,) target
    Returns:
      beta: (p,) LS params
      se:   (p,) standard errors
      ci:   (p, 2) [lo, hi] Student-t CI
      cov:  (p, p) covariance of beta
      df:   degrees of freedom (n - p)
    """
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = max(n - p, 1)
    s2 = float(resid @ resid) / df
    XtX = X.T @ X
    cov = s2 * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(cov))
    tcrit = t_dist.ppf(1 - alpha / 2, df)
    ci = np.column_stack([beta - tcrit * se, beta + tcrit * se])
    return beta, se, ci, cov, df


def ci_halfwidth_pct(est, lo, hi, eps=1e-12):
    """Return half-width of CI as a percentage of |est|."""
    hw = 0.5 * (hi - lo)
    denom = max(abs(float(est)), eps)
    return 100.0 * hw / denom


def euler_rates_from_body_rates(phi, theta, p, q, r, eps=1e-6):
    """
    Convert body rates (p,q,r) to Euler angle rates for ZYX (yaw-pitch-roll).
    Returns: phi_dot, theta_dot, psi_dot
    """
    ct = np.cos(theta)
    ct = np.clip(ct, eps, None)  # avoid division by ~0 (pitch ~ ±90°)
    st = np.sin(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    tt = st / ct  # tan(theta)
    sec = 1.0 / ct

    phi_dot = p + q * sp * tt + r * cp * tt
    theta_dot = q * cp - r * sp
    psi_dot = (q * sp + r * cp) * sec
    return phi_dot, theta_dot, psi_dot


def Rmat_batch(phi, th, psi):
    c, s = np.cos, np.sin
    Rx = lambda a: np.stack(
        [
            np.ones_like(a),
            np.zeros_like(a),
            np.zeros_like(a),
            np.zeros_like(a),
            c(a),
            -s(a),
            np.zeros_like(a),
            s(a),
            c(a),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    Ry = lambda a: np.stack(
        [
            c(a),
            np.zeros_like(a),
            s(a),
            np.zeros_like(a),
            np.ones_like(a),
            np.zeros_like(a),
            -s(a),
            np.zeros_like(a),
            c(a),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    Rz = lambda a: np.stack(
        [
            c(a),
            -s(a),
            np.zeros_like(a),
            s(a),
            c(a),
            np.zeros_like(a),
            np.zeros_like(a),
            np.zeros_like(a),
            np.ones_like(a),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    return Rz(psi) @ Ry(th) @ Rx(phi)


def filter_flight_data(
    data, cutoff_acc=30, cutoff_rates=40, cutoff_att=40, cutoff_motor=60
):
    """
    Apply 4th-order low-pass Butterworth filters to accelerations, angular rates, and motor speeds.
    """
    import scipy.signal as sig

    fs = 1.0 / np.mean(np.diff(data["t"]))

    def butter_filter(signal, cutoff):
        sos = sig.butter(4, cutoff, "low", fs=fs, output="sos")
        return sig.sosfiltfilt(sos, signal)

    # Filter accelerometer data
    for k in ["ax", "ay", "az", "ax_ned", "ay_ned", "az_ned"]:
        if k in data:
            data[f"{k}_filt"] = butter_filter(data[k], cutoff_acc)

    # Filter gyroscope data
    for k in ["p", "q", "r"]:
        if k in data:
            data[f"{k}_filt"] = butter_filter(data[k], cutoff_rates)

    # Filter motor speeds
    for i in range(4):
        key = f"omega[{i}]"
        if key in data:
            data[f"{key}_filt"] = butter_filter(data[key], cutoff_motor)

    # Filter euler angles (optional – useful for other plots)
    for k in ["phi", "theta"]:
        if k in data:
            data[f"{k}_filt"] = butter_filter(
                data[k], cutoff_att
            )  # reuse gyro cutoff or choose separate

    return data


def load_paparazzi_log(fname, rpm_min=2500, rpm_max=12000):
    # --- robust CSV reader (handles BOMs, uneven rows, different delimiters) ---
    bad_rows = 0
    rows = []

    with open(fname, "r", encoding="utf-8-sig", newline="") as f:
        # sniff delimiter on a small sample (fallback to comma)
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.excel
            dialect.delimiter = ","

        reader = csv.reader(f, dialect)
        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError(f"{fname} appears to be empty")

        # strip whitespace and BOM remnants from header names
        header = [h.strip().strip("\ufeff") for h in header]
        n_cols = len(header)

        for line_idx, row in enumerate(reader, start=2):
            # skip comments/blank lines
            if not row or (len(row) == 1 and row[0].strip() == ""):
                continue

            # some loggers put trailing delimiter -> empty last field; normalize row length
            if len(row) < n_cols:
                # pad missing cells with empty strings
                row = row + [""] * (n_cols - len(row))
            elif len(row) > n_cols:
                # too many fields → treat as malformed row and skip
                bad_rows += 1
                continue

            # numeric coercion with NaN fallback
            parsed = []
            for x in row:
                xs = x.strip()
                if xs == "" or xs.lower() in ("nan", "inf", "-inf"):
                    parsed.append(np.nan)
                else:
                    try:
                        parsed.append(float(xs))
                    except ValueError:
                        # non-numeric token (e.g., JSON/list/etc.) → NaN
                        parsed.append(np.nan)
            rows.append(parsed)

    if not rows:
        raise RuntimeError(
            f"No valid data rows parsed from {fname}. "
            f"Header had {n_cols} columns; skipped {bad_rows} malformed rows."
        )

    if bad_rows:
        print(f"[loader] Skipped {bad_rows} malformed row(s) (field-count mismatch).")

    raw = np.array(rows, dtype=float)
    col = {name: i for i, name in enumerate(header)}

    data = {}

    # --- time ---
    data["t"] = raw[:, col["time"]] - raw[0, col["time"]]

    # --- attitude ---
    data["phi"] = raw[:, col["att_phi"]]
    data["theta"] = raw[:, col["att_theta"]]
    data["psi"] = raw[:, col["att_psi"]]

    R = Rmat_batch(data["phi"], data["theta"], data["psi"])

    # --- position / velocity (ENU→NED reorder) ---
    enu2ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    pos_enu = raw[:, [col["pos_x"], col["pos_y"], col["pos_z"]]]
    vel_enu = raw[:, [col["vel_x"], col["vel_y"], col["vel_z"]]]
    pos_ned = pos_enu @ enu2ned.T
    vel_ned = vel_enu @ enu2ned.T
    data["x"], data["y"], data["z"] = pos_ned.T
    data["vx"], data["vy"], data["vz"] = vel_ned.T

    # --- body-frame velocities (for drag fit) ---
    data["vbx"], data["vby"], data["vbz"] = (
        np.einsum("nij,nj->ni", R.transpose(0, 2, 1), vel_ned)
    ).T
    data["v"] = np.linalg.norm([data["vx"], data["vy"], data["vz"]], axis=0)

    # --- acceleration NED → body FRD ---
    acc_ned = raw[:, [col["acc_x"], col["acc_y"], col["acc_z"]]]

    acc_ned[:, 2] -= 9.81  #!!! Add gravity

    # store the raw NED values *before* rotating
    data["ax_ned"], data["ay_ned"], data["az_ned"] = acc_ned.T
    # rotate to body frame
    data["ax"], data["ay"], data["az"] = (
        np.einsum("nij,nj->ni", R.transpose(0, 2, 1), acc_ned)
    ).T

    # --- gyro body (already FRD) ---
    data["p"] = raw[:, col["rate_p"]]
    data["q"] = raw[:, col["rate_q"]]
    data["r"] = raw[:, col["rate_r"]]

    # OPTIONAL -- map CTBR-INDI command columns if they exist from neural controller
    #      nn_p,  nn_q,  nn_r,  nn_thrust   ->  p_cmd, q_cmd, r_cmd, T_cmd
    #      cmd_yaw                           ->  r_cmd   (over-rules nn_r if both exist)
    # ----------------------------------------------------------------------------
    cmd_map = {
        "nn_p": "p_cmd",
        "nn_q": "q_cmd",
        "nn_r": "r_cmd",
        "nn_thrust": "T_cmd",
    }

    for src, dst in cmd_map.items():
        if src in col:  # present in this log → copy it
            data[dst] = raw[:, col[src]].astype(float)

    # --- motor speeds & commands ---
    if f"rpm_obs_1" in col:
        for i in range(4):
            data[f"omega[{i}]"] = raw[:, col[f"rpm_obs_{i+1}"]] * 2 * np.pi / 60
    have_ref = f"rpm_ref_1" in col
    if have_ref:
        for i in range(4):
            ui = (raw[:, col[f"rpm_ref_{i+1}"]] - rpm_min) / (rpm_max - rpm_min)
            data[f"u{i+1}"] = np.clip(ui, 0, 1)

    cmd_cols = ["accel_command_x", "accel_command_y", "accel_command_z"]
    if all(name in col for name in cmd_cols):
        data["ax_cmd_ned"] = raw[:, col["accel_command_x"]].astype(float)
        data["ay_cmd_ned"] = raw[:, col["accel_command_y"]].astype(float)
        data["az_cmd_ned"] = raw[:, col["accel_command_z"]].astype(float)

    data["az_cmd_ned"] -= 9.81  # need to correct for gravity

    # --- attitude commanded (only if guidance_indi controller is active) ---
    if f"cmd_euler_phi" in col:
        data["cmd_euler_phi"] = raw[:, col["cmd_euler_phi"]].astype(float)
    if f"cmd_euler_theta" in col:
        data["cmd_euler_theta"] = raw[:, col["cmd_euler_theta"]].astype(float)

    # (Optional) if your log stores degrees, convert to rad here:
    # data["cmd_euler_phi"]   = np.deg2rad(data["cmd_euler_phi"])
    # data["cmd_euler_theta"] = np.deg2rad(data["cmd_euler_theta"])

    return data


def fit_thrust_drag_model(data):
    print("fitting thrust and drag model")
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=False)

    # THRUST MODEL ------------------------------------------------------------------------------
    # az = k_w*sum(omega_i**2)
    if PRINT_CI:
        X = (
            data["omega[0]"] ** 2
            + data["omega[1]"] ** 2
            + data["omega[2]"] ** 2
            + data["omega[3]"] ** 2
        )
        Y = data["az_filt"]
        (k_w,), se, ci, cov, df = ols_with_ci(X, Y)
        print(f"k_w = {k_w:.5e}  95% CI = [{ci[0,0]:.5e}, {ci[0,1]:.5e}]  (df={df})")
        pct = ci_halfwidth_pct(k_w, ci[0, 0], ci[0, 1])
        print(f"    (±{pct:.2f}% half-width)")

    X = np.stack(
        [
            data["omega[0]"] ** 2
            + data["omega[1]"] ** 2
            + data["omega[2]"] ** 2
            + data["omega[3]"] ** 2,
            # data['vbx']**2 + data['vby']**2,
            # data['vz']*(data['omega[0]']+data['omega[1]']+data['omega[2]']+data['omega[3]'])
        ]
    )
    Y = data["az_filt"]
    (k_w,) = A = np.linalg.lstsq(X.T, Y, rcond=None)[0]

    # if "az_unfiltered" in data:
    #     axs[0].plot(
    #         data["t"], data["az_unfiltered"], label="az raw", alpha=0.1, color="blue"
    #     )
    axs[0].plot(data["t"], Y, label="az")  # , alpha=0.2)
    axs[0].plot(data["t"], data["az_filt"], label="az filt")
    axs[0].plot(data["t"], A @ X, label="T model")
    # axs[0].plot(data['t'], A_nom@X, label='T model nominal')
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("acc [m/s^2]")
    axs[0].legend()
    axs[0].set_title("Thrust model: az = k_w*sum(omega_i**2) k_w = {:.2e}".format(k_w))
    # axs[0].set_title('Thrust model: \n az = k_w*sum(omega_i**2) + k_h*(vbx**2+vby**2) + k_z*vbz*sum(omega_i) \n k_w, k_h, k_z = {:.2e}, {:.2e}, {:.2e}'.format(k_w, k_h, k_z))

    # DRAG MODEL X ------------------------------------------------------------------------------
    # Eq. 2 from https://doi.org/10.1016/j.robot.2023.104588
    # ax = -k_x*vbx*sum(omega_i)
    if PRINT_CI:
        X = data["vbx"] * (
            data["omega[0]"] + data["omega[1]"] + data["omega[2]"] + data["omega[3]"]
        )
        Y = data["ax_filt"]
        (k_x_motor,), se, ci, cov, df = ols_with_ci(X, Y)
        print(f"k_x_motor = {k_x_motor:.5e}  95% CI = [{ci[0,0]:.5e}, {ci[0,1]:.5e}]")
        pct = ci_halfwidth_pct(k_x_motor, ci[0, 0], ci[0, 1])
        print(f"    (±{pct:.2f}% half-width)")

    X = np.stack(
        [
            data["vbx"]
            * (
                data["omega[0]"]
                + data["omega[1]"]
                + data["omega[2]"]
                + data["omega[3]"]
            )
        ]
    )
    # X = np.stack([data['vbx']])
    Y = data["ax_filt"]
    (k_x,) = A = np.linalg.lstsq(X.T, Y, rcond=None)[0]

    # if "ax_unfiltered" in data:
    #     axs[1].plot(
    #         data["t"], data["ax_unfiltered"], label="ax raw", alpha=0.1, color="blue"
    #     )
    axs[1].plot(data["t"], Y, label="ax")  # , alpha=0.2)
    axs[1].plot(data["t"], data["ax_filt"], label="ax filt")
    axs[1].plot(data["t"], A @ X, label="Dx model")
    # axs[1].plot(data['t'], A_nom@X, label='Dx model nominal')
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("acc [m/s^2]")
    axs[1].legend()
    axs[1].set_title("Drag model X: ax = k_x*vbx*sum(omega_i) k_x = {:.2e}".format(k_x))

    # DRAG MODEL Y ------------------------------------------------------------------------------
    # Eq. 2 from https://doi.org/10.1016/j.robot.2023.104588
    # ay = -k_y*vby*sum(omega_i)
    if PRINT_CI:
        X = data["vby"] * (
            data["omega[0]"] + data["omega[1]"] + data["omega[2]"] + data["omega[3]"]
        )
        Y = data["ay_filt"]
        (k_y_motor,), se, ci, cov, df = ols_with_ci(X, Y)
        print(f"k_y_motor = {k_y_motor:.5e}  95% CI = [{ci[0,0]:.5e}, {ci[0,1]:.5e}]")
        pct = ci_halfwidth_pct(k_y_motor, ci[0, 0], ci[0, 1])
        print(f"    (±{pct:.2f}% half-width)")

    X = np.stack(
        [
            data["vby"]
            * (
                data["omega[0]"]
                + data["omega[1]"]
                + data["omega[2]"]
                + data["omega[3]"]
            )
        ]
    )
    # X = np.stack([data['vby']])
    Y = data["ay_filt"]
    (k_y,) = A = np.linalg.lstsq(X.T, Y, rcond=None)[0]

    # if "ay_unfiltered" in data:
    #     axs[2].plot(
    #         data["t"], data["ay_unfiltered"], label="ay raw", alpha=0.1, color="blue"
    #     )
    axs[2].plot(data["t"], Y, label="ay")  # , alpha=0.2)
    axs[2].plot(data["t"], data["ay_filt"], label="ay filt")
    axs[2].plot(data["t"], A @ X, label="Dy model")
    # axs[2].plot(data['t'], A_nom@X, label='Dy model nominal')
    axs[2].set_xlabel("t [s]")
    axs[2].set_ylabel("acc [m/s^2]")
    axs[2].legend()
    axs[2].set_title("Drag model Y: ay = k_y*vby*sum(omega_i) k_y = {:.2e}".format(k_y))
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Thrust and Drag Model")
    plt.show()

    # print('k_w = {:.2e}, k_x = {:.2e}, k_y = {:.2e}'.format(k_w, k_x, k_y))
    return k_w, k_x, k_y


def fit_actuator_model(data):
    # the steadystate rpm motor response to the motor command u is described by:
    # w_c = (w_max-w_min)*sqrt(k u**2 + (1-k)*u) + w_min
    # the dynamics of the motor is described by:
    # dw/dt = (w_c - w)/tau
    # dw/dt = ((w_max-w_min)*sqrt(k u**2 + (1-k)*u) + w_min - w)*tau_inv
    # we will find w_min, w_max, k, tau_inv by nonlinear optimization

    def get_w_est(params, u, w):
        w_min, w_max, k, tau_inv = params
        w_c = (w_max - w_min) * np.sqrt(k * u**2 + (1 - k) * u) + w_min
        # progate the dynamics
        w_est = np.zeros_like(u)
        w_est[0] = w[0]
        for i in range(1, len(w_est)):
            dt = data["t"][i] - data["t"][i - 1]
            w_est[i] = w_est[i - 1] + (w_c[i] - w_est[i - 1]) * dt * tau_inv
        return w_est

    def get_error(params, u, w):
        return np.linalg.norm(get_w_est(params, u, w) - w)

    # w_min, w_max, k, tau_inv
    initial_guess = [300, 1200, 0.85, 100]
    bounds = [(0, 1000), (0, 6000), (0, 1), (1, 1000.0)]

    # minimize for each motor
    err_1 = lambda x: get_error(x, data["u1"], data["omega[0]"])
    err_2 = lambda x: get_error(x, data["u2"], data["omega[1]"])
    err_3 = lambda x: get_error(x, data["u3"], data["omega[2]"])
    err_4 = lambda x: get_error(x, data["u4"], data["omega[3]"])
    err_tot = lambda x: err_1(x) + err_2(x) + err_3(x) + err_4(x)

    print("fitting actuator model...")
    res_1 = minimize(err_1, initial_guess, bounds=bounds)
    res_2 = minimize(err_2, initial_guess, bounds=bounds)
    res_3 = minimize(err_3, initial_guess, bounds=bounds)
    res_4 = minimize(err_4, initial_guess, bounds=bounds)
    res_tot = minimize(err_tot, initial_guess, bounds=bounds)

    # set k to 45
    # res_tot.x[2] = 0.45

    # plot results
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    axs[0, 0].plot(data["t"], data["omega[0]"], label="w")
    axs[0, 0].plot(
        data["t"], get_w_est(res_1.x, data["u1"], data["omega[0]"]), label="w est"
    )
    axs[0, 0].plot(
        data["t"], get_w_est(res_tot.x, data["u1"], data["omega[0]"]), label="w est tot"
    )
    # axs[0,0].plot(data['t'], get_w_est(res_nom, data['u1'], data['omega[0]']), label='w est nom')
    axs[0, 0].set_xlabel("t [s]")
    axs[0, 0].set_ylabel("w [rad/s]")
    axs[0, 0].legend()
    params = res_1.x
    params[3] = 1 / params[3]
    axs[0, 0].set_title(
        "Motor 1: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, tau = {:.2f}".format(
            *params
        )
    )

    axs[0, 1].plot(data["t"], data["omega[1]"], label="w")
    axs[0, 1].plot(
        data["t"], get_w_est(res_2.x, data["u2"], data["omega[1]"]), label="w_est"
    )
    axs[0, 1].plot(
        data["t"], get_w_est(res_tot.x, data["u2"], data["omega[1]"]), label="w est tot"
    )
    # axs[0,1].plot(data['t'], get_w_est(res_nom, data['u2'], data['omega[1]']), label='w est nom')
    axs[0, 1].set_xlabel("t [s]")
    axs[0, 1].set_ylabel("w [rad/s]")
    axs[0, 1].legend()
    params = res_2.x
    params[3] = 1 / params[3]
    axs[0, 1].set_title(
        "Motor 2: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, tau = {:.2f}".format(
            *params
        )
    )

    axs[1, 0].plot(data["t"], data["omega[2]"], label="w")
    axs[1, 0].plot(
        data["t"], get_w_est(res_3.x, data["u3"], data["omega[2]"]), label="w_est"
    )
    axs[1, 0].plot(
        data["t"], get_w_est(res_tot.x, data["u3"], data["omega[2]"]), label="w est tot"
    )
    # axs[1,0].plot(data['t'], get_w_est(res_nom, data['u3'], data['omega[2]']), label='w est nom')
    axs[1, 0].set_xlabel("t [s]")
    axs[1, 0].set_ylabel("w [rad/s]")
    axs[1, 0].legend()
    params = res_3.x
    params[3] = 1 / params[3]
    axs[1, 0].set_title(
        "Motor 3: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, tau = {:.2f}".format(
            *params
        )
    )

    axs[1, 1].plot(data["t"], data["omega[3]"], label="w")
    axs[1, 1].plot(
        data["t"], get_w_est(res_4.x, data["u4"], data["omega[3]"]), label="w_est"
    )
    axs[1, 1].plot(
        data["t"], get_w_est(res_tot.x, data["u4"], data["omega[3]"]), label="w est tot"
    )
    # axs[1,1].plot(data['t'], get_w_est(res_nom, data['u4'], data['omega[3]']), label='w est nom')
    axs[1, 1].set_xlabel("t [s]")
    axs[1, 1].set_ylabel("w [rad/s]")
    axs[1, 1].legend()
    params = res_4.x
    params[3] = 1 / params[3]
    axs[1, 1].set_title(
        "Motor 4: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, tau = {:.2f}".format(
            *params
        )
    )

    # suptitle
    params = res_tot.x
    params[3] = 1 / params[3]
    fig.suptitle(
        "Actuator model: \n dw/dt = dw/dt = ((w_max-w_min)*sqrt(k u**2 + (1-k)*u) + w_min - w)/tau \n Total fit: w_min = {:.2f}, w_max = {:.2f}, k = {:.2f}, 1/tau = {:.2f}".format(
            *params
        )
    )

    # show fig with the window name 'Actuator Model'
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Actuator Model")
    plt.show()

    # print('w_min={:.2f}, w_max={:.2f}, k={:.2f}, tau={:.2f}'.format(*res_tot.x))
    return res_tot.x


def fit_actuator_model_with_ci(data, alpha=0.05):
    # params: [w_min, w_max, k, tau_inv]
    def get_w_est(params, u, w, t):
        w_min, w_max, k, tau_inv = params
        w_c = (w_max - w_min) * np.sqrt(k * u**2 + (1 - k) * u) + w_min
        w_est = np.zeros_like(u)
        w_est[0] = w[0]
        for i in range(1, len(w_est)):
            dt = t[i] - t[i - 1]
            w_est[i] = w_est[i - 1] + (w_c[i] - w_est[i - 1]) * dt * tau_inv
        return w_est

    t = data["t"]
    u = [data["u1"], data["u2"], data["u3"], data["u4"]]
    w = [data["omega[0]"], data["omega[1]"], data["omega[2]"], data["omega[3]"]]

    def residuals(p):
        r = [get_w_est(p, ui, wi, t) - wi for ui, wi in zip(u, w)]
        return np.concatenate(r)

    x0 = np.array([300.0, 6000.0, 0.85, 100.0])
    bounds = ([0.0, 0.0, 0.0, 1.0], [1000.0, 20000.0, 1.0, 2000.0])
    res = least_squares(residuals, x0, bounds=bounds, method="trf", jac="2-point")

    # covariance via (J^T J)^{-1} scaled by residual variance
    J = res.jac  # (N x p)
    r = res.fun
    n, p = J.shape
    df = max(n - p, 1)
    s2 = float(r @ r) / df
    cov = s2 * np.linalg.inv(J.T @ J)
    se = np.sqrt(np.diag(cov))
    tcr = t_dist.ppf(1 - alpha / 2, df)
    ci = np.column_stack([res.x - tcr * se, res.x + tcr * se])

    # nice printout (also report tau instead of tau_inv if you prefer)
    names = ["w_min", "w_max", "k", "tau_inv"]
    for name, val, (lo, hi) in zip(names, res.x, ci):
        print(f"{name:7s} = {val:10.4f}   95% CI [{lo:.4f}, {hi:.4f}]  (df={df})")

    # If you want CI for tau instead of tau_inv, delta method:
    tau = 1.0 / res.x[3]
    var_tau = cov[3, 3] / (res.x[3] ** 4)  # (d(1/x)/dx = -1/x^2)
    se_tau = np.sqrt(var_tau)
    tau_ci = (tau - tcr * se_tau, tau + tcr * se_tau)
    tau_pct = ci_halfwidth_pct(tau, tau_ci[0], tau_ci[1])
    print(
        f"tau     = {tau:.6f} s   95% CI [{tau_ci[0]:.6f}, {tau_ci[1]:.6f}]  (±{tau_pct:.2f}% hw)"
    )

    for name, val, (lo, hi) in zip(names, res.x, ci):
        pct = ci_halfwidth_pct(val, lo, hi)
        print(
            f"{name:7s} = {val:10.4f}   95% CI [{lo:.4f}, {hi:.4f}]  (±{pct:.2f}% hw)  (df={df})"
        )

    return res.x, ci, cov, res


def fit_moments_model(data):
    print("fitting moments model")
    # model from https://doi.org/10.1016/j.robot.2023.104588
    # d_p     = (q*r*(Iyy-Izz) + Mx)/Ixx = Jx*q*r + Mx_
    # d_q     = (p*r*(Izz-Ixx) + My)/Iyy = Jy*p*r + My_
    # d_r     = (p*q*(Ixx-Iyy) + Mz)/Izz = Jz*p*q + Mz_

    # where
    # Mx_ = k_p1*omega_1**2 + k_p2*omega_2**2 + k_p3*omega_3**2 + k_p4*omega_4**2
    # My_ = k_q1*omega_1**2 + k_q2*omega_2**2 + k_q3*omega_3**2 + k_q4*omega_4**2
    # Mz_ = k_r1*omega_1 + k_r2*omega_2 + k_r3*omega_3 + k_r4*omega_4 + k_r5*d_omega_1 + k_r6*d_omega_2 + k_r7*d_omega_3 + k_r8*d_omega_4

    dp = np.gradient(data["p_filt"]) / np.gradient(data["t"])
    dq = np.gradient(data["q_filt"]) / np.gradient(data["t"])
    dr = np.gradient(data["r_filt"]) / np.gradient(data["t"])

    domega_1 = np.gradient(data["omega[0]_filt"]) / np.gradient(data["t"])
    domega_2 = np.gradient(data["omega[1]_filt"]) / np.gradient(data["t"])
    domega_3 = np.gradient(data["omega[2]_filt"]) / np.gradient(data["t"])
    domega_4 = np.gradient(data["omega[3]_filt"]) / np.gradient(data["t"])

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)

    X = np.stack(
        [
            data["omega[0]"] ** 2,
            data["omega[1]"] ** 2,
            data["omega[2]"] ** 2,
            data["omega[3]"] ** 2,
        ]
    )
    Y = dp
    A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    k_p1, k_p2, k_p3, k_p4 = A
    dp_fit = A @ X

    axs[0].plot(data["t"], Y, label="dp")
    axs[0].plot(data["t"], dp_fit, label="dp fit")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("dp [rad/s^2]")
    axs[0].legend()
    axs[0].set_title(
        "dp = k_p1*w1**2 + k_p2*w2**2 + k_p3*w3**2 + k_p4*w4**2 \n k_p1, k_p2, k_p3, k_p4 = {:.2e}, {:.2e}, {:.2e}, {:.2e}".format(
            *A
        )
    )

    X = np.stack(
        [
            data["omega[0]"] ** 2,
            data["omega[1]"] ** 2,
            data["omega[2]"] ** 2,
            data["omega[3]"] ** 2,
        ]
    )
    Y = dq
    A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    k_q1, k_q2, k_q3, k_q4 = A
    dq_fit = A @ X

    axs[1].plot(data["t"], Y, label="dq")
    axs[1].plot(data["t"], dq_fit, label="dq fit")
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("dq [rad/s^2]")
    axs[1].legend()
    axs[1].set_title(
        "dq = k_q1*w1**2 + k_q2*w2**2 + k_q3*w3**2 + k_q4*w4**2 \n k_q1, k_q2, k_q3, k_q4 = {:.2e}, {:.2e}, {:.2e}, {:.2e}".format(
            *A
        )
    )

    X = np.stack(
        [
            data["omega[0]"],
            data["omega[1]"],
            data["omega[2]"],
            data["omega[3]"],
            domega_1,
            domega_2,
            domega_3,
            domega_4,
        ]
    )
    Y = dr
    A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8 = A
    dr_fit = A @ X

    # X = np.stack(
    #     [
    #         -data["omega[0]"] + data["omega[1]"] + data["omega[2]"] - data["omega[3]"],
    #         -domega_1 + domega_2 + domega_3 - domega_4,
    #     ]
    # )
    # Y = dr
    # A = np.linalg.lstsq(X.T, Y, rcond=None)[0]
    # k_r, k_rd = A
    # dr_fit = A @ X

    axs[2].plot(data["t"], Y, label="dr")
    # axs[2,0].plot(data['t'], dr_fit, label='dr fit')
    axs[2].plot(data["t"], dr_fit, label="dr fit")
    axs[2].set_xlabel("t [s]")
    axs[2].set_ylabel("dr [rad/s^2]")
    axs[2].legend()
    title = "dr = k_r1*w1 + k_r2*w2 + k_r3*w3 + k_r4*w4 + k_r5*dw1 + k_r6*dw2 + k_r7*dw3 + k_r8*dw4 \n k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8 = {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}".format(
        *A
    )
    # title = "dr = k_r*(w1-w2+w3-w4) + k_rd*(dw1-dw2+dw3-dw4) \n k_r, k_rd = {:.2e}, {:.2e}".format(
    #     *A
    # )
    axs[2].set_title(title)

    # 3 plots with p,q,r
    # axs[0, 1].plot(data["t"], dp_fit - dp, label="dp fit error")
    # axs[0, 1].set_xlabel("t [s]")
    # axs[0, 1].set_ylabel("p [rad/s]")
    # axs[0, 1].legend()

    # axs[1, 1].plot(data["t"], dq_fit - dq, label="dq fit error")
    # axs[1, 1].set_xlabel("t [s]")
    # axs[1, 1].set_ylabel("q [rad/s]")
    # axs[1, 1].legend()

    # axs[2, 1].plot(data["t"], dr_fit - dr, label="dr fit error")
    # axs[2, 1].set_xlabel("t [s]")
    # axs[2, 1].set_ylabel("r [rad/s]")
    # axs[2, 1].legend()

    # show fig with the window name 'Moments Model'
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Moments Model")
    plt.show()

    # print the results
    # print('k_p1, k_p2, k_p3, k_p4 = {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(k_p1, k_p2, k_p3, k_p4))
    # print('k_q1, k_q2, k_q3, k_q4 = {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(k_q1, k_q2, k_q3, k_q4))
    # print('k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8 = {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8))

    if PRINT_CI:
        # dp fit
        X = np.stack(
            [
                data["omega[0]"] ** 2,
                data["omega[1]"] ** 2,
                data["omega[2]"] ** 2,
                data["omega[3]"] ** 2,
            ],
            axis=1,
        )
        Y = np.gradient(data["p_filt"]) / np.gradient(data["t"])
        beta, se, ci, cov, df = ols_with_ci(X, Y)
        k_p1, k_p2, k_p3, k_p4 = beta
        print("dp coeffs (95% CI each):")
        for name, b, (lo, hi) in zip(["k_p1", "k_p2", "k_p3", "k_p4"], beta, ci):
            pct = ci_halfwidth_pct(b, lo, hi)
            print(f"  {name} = {b:.3e}  [{lo:.3e}, {hi:.3e}]  (±{pct:.2f}% hw)")

        # dq fit (same X, new Y)
        Y = np.gradient(data["q_filt"]) / np.gradient(data["t"])
        beta, se, ci, cov, df = ols_with_ci(X, Y)
        k_q1, k_q2, k_q3, k_q4 = beta
        print("dq coeffs (95% CI each):")
        for name, b, (lo, hi) in zip(["k_q1", "k_q2", "k_q3", "k_q4"], beta, ci):
            pct = ci_halfwidth_pct(b, lo, hi)
            print(f"  {name} = {b:.3e}  [{lo:.3e}, {hi:.3e}]  (±{pct:.2f}% hw)")

        # dr fit (linear in omegas and d(omega))
        X = np.stack(
            [
                data["omega[0]"],
                data["omega[1]"],
                data["omega[2]"],
                data["omega[3]"],
                np.gradient(data["omega[0]_filt"]) / np.gradient(data["t"]),
                np.gradient(data["omega[1]_filt"]) / np.gradient(data["t"]),
                np.gradient(data["omega[2]_filt"]) / np.gradient(data["t"]),
                np.gradient(data["omega[3]_filt"]) / np.gradient(data["t"]),
            ],
            axis=1,
        )
        Y = np.gradient(data["r_filt"]) / np.gradient(data["t"])
        beta, se, ci, cov, df = ols_with_ci(X, Y)
        k_r1, k_r2, k_r3, k_r4, k_r5, k_r6, k_r7, k_r8 = beta
        print("dr coeffs (95% CI each):")
        for name, b, (lo, hi) in zip([f"k_r{i}" for i in range(1, 9)], beta, ci):
            pct = ci_halfwidth_pct(b, lo, hi)
            print(f"  {name} = {b:.3e}  [{lo:.3e}, {hi:.3e}]  (±{pct:.2f}% hw)")

    return (
        k_p1,
        k_p2,
        k_p3,
        k_p4,
        k_q1,
        k_q2,
        k_q3,
        k_q4,
        k_r1,
        k_r2,
        k_r3,
        k_r4,
        k_r5,
        k_r6,
        k_r7,
        k_r8,
    )


def plot_psd(data, fmin=0.1, fmax=200, nperseg=2560 * 2):
    fs = 1.0 / np.mean(np.diff(data["t"]))  # Sampling frequency
    acc_keys = ["ax_ned", "ay_ned", "az_ned"]
    rate_keys = ["p", "q", "r"]
    motor_keys = [f"omega[{i}]" for i in range(4)]

    fig, axs = plt.subplots(1, 1, figsize=(10, 12), sharex=True, sharey=True)

    # Accelerometer PSD
    # for k in acc_keys:
    #     f, Pxx = sig.welch(data[k], fs=fs, nperseg=nperseg)
    #     axs.plot(f, 10 * np.log10(Pxx), label=k)
    # axs.set_ylabel("PSD [dB/Hz]")
    # axs.set_title("Accelerometer PSD (Body Frame)")
    # axs.grid(True, which="both", ls="--", alpha=0.5)
    # axs.legend()

    # Gyroscope PSD
    for k in rate_keys:
        f, Pxx = sig.welch(data[k], fs=fs, nperseg=nperseg)
        axs.plot(f, 10 * np.log10(Pxx), label=k)
    axs.set_ylabel("PSD [dB/Hz]")
    axs.set_title("Gyroscope PSD")
    axs.grid(True, which="both", ls="--", alpha=0.5)
    axs.legend()

    # # Motor speed PSD
    # for k in motor_keys:
    #     f, Pxx = sig.welch(data[k], fs=fs, nperseg=nperseg)
    #     axs[2].plot(f, 10 * np.log10(Pxx), label=k)
    # axs[2].set_ylabel("PSD [dB/Hz]")
    # axs[2].set_xlabel("Frequency [Hz]")
    # axs[2].set_title("Motor Speed PSD")
    # axs[2].grid(True, which="both", ls="--", alpha=0.5)
    # axs[2].legend()

    # Log-scale frequency axis
    # for ax in axs:
    axs.set_xscale("log")
    axs.set_xlim(0.1, 500)
    axs.set_ylim(-60, 15)

    plt.tight_layout()
    plt.show()


# ------------------------------ CTBR SYSTEM IDENTIFICATION ------------------------------


def _linreg_1d(x, y):
    return np.linalg.lstsq(x[:, None], y, rcond=None)[0][0]


def fit_ctbr_planar_drag(data, v_thresh=0.0):
    """
    Estimate the planar body-drag coefficients used in CTBR-INDI:
        D_x = −k_x · v_bx
        D_y = −k_y · v_by
    """
    # use filtered acc to suppress noise
    ax, ay = data["ax_filt"], data["ay_filt"]
    vbx, vby = data["vbx"], data["vby"]

    # ignore samples where the regressor is too small (singular)
    mask_x = np.abs(vbx) > v_thresh
    mask_y = np.abs(vby) > v_thresh

    k_x = -_linreg_1d(vbx[mask_x], ax[mask_x])
    k_y = -_linreg_1d(vby[mask_y], ay[mask_y])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, (v, a, k, lbl) in enumerate(
        [
            (data["vbx"], data["ax_filt"], k_x, "X-body"),
            (data["vby"], data["ay_filt"], k_y, "Y-body"),
        ]
    ):
        m = np.abs(v) > v_thresh
        ax[i].scatter(v[m], a[m], s=8, alpha=0.4, label="samples")
        v_line = np.linspace(v[m].min(), v[m].max(), 200)
        ax[i].plot(v_line, -k * v_line, "r", lw=2, label=f"fit  (k={k:.2e})")
        ax[i].set_xlabel(f"v_b{lbl[0].lower()}  [m/s]")
        ax[i].set_ylabel(f"a_{lbl[0].lower()}  [m/s²]")
        ax[i].set_title(f"Planar drag – {lbl}")
        ax[i].grid(True, ls="--", alpha=0.3)
        ax[i].legend()
    fig.tight_layout()
    plt.show()

    if PRINT_CI:
        # X-body
        m = np.abs(data["vbx"]) > 0.1
        beta, se, ci, cov, df = ols_with_ci(data["vbx"][m], data["ax_filt"][m])
        k_x = -beta[0]
        ci_kx = np.sort(-ci[0])  # flip because of the minus sign
        print(f"k_x = {k_x:.3e}  95% CI = [{ci_kx[0]:.3e}, {ci_kx[1]:.3e}]")
        pct = ci_halfwidth_pct(k_x, ci_kx[0], ci_kx[1])
        print(f"    (±{pct:.2f}% half-width)")

        # Y-body
        m = np.abs(data["vby"]) > 0.1
        beta, se, ci, cov, df = ols_with_ci(data["vby"][m], data["ay_filt"][m])
        k_y = -beta[0]
        ci_ky = np.sort(-ci[0])
        print(f"k_y = {k_y:.3e}  95% CI = [{ci_ky[0]:.3e}, {ci_ky[1]:.3e}]")
        pct = ci_halfwidth_pct(k_y, ci_ky[0], ci_ky[1])
        print(f"    (±{pct:.2f}% half-width)")

    return k_x, k_y


def tau_and_ci_from_channel(y, y_cmd, t, alpha=0.05):
    dy = np.gradient(y, t)
    X = y_cmd - y
    m = np.abs(X) > 1e-3
    (tau_inv,), se, ci, cov, df = ols_with_ci(X[m], dy[m], alpha=alpha)

    # delta method: Var(1/x) ≈ (1/x^4) Var(x)
    tau = 1.0 / tau_inv
    var = cov[0, 0] / (tau_inv**4)
    se_tau = np.sqrt(var)
    tcr = t_dist.ppf(1 - alpha / 2, df)
    ci_tau = (tau - tcr * se_tau, tau + tcr * se_tau)
    tau_pct = ci_halfwidth_pct(tau, ci_tau[0], ci_tau[1])
    print(
        f"tau = {tau:.6f} s   95% CI [{ci_tau[0]:.6f}, {ci_tau[1]:.6f}]  (±{tau_pct:.2f}% hw)"
    )

    return tau, ci_tau, tau_inv, (ci[0, 0], ci[0, 1]), df


def estimate_time_constants_ctbr(data):
    """
    Estimate actuator time constants for p, q, r, and T_norm (from -az_filt) using:
        dy/dt = (y_cmd - y) / tau
        => tau = least-squares fit of (y_cmd - y) to dy/dt

    Plots the simulated vs measured response using the fitted time constant.
    """
    t = data["t"]
    taus = {}

    def estimate_tau(y, y_cmd, label):
        dy = np.gradient(y, t)
        X = y_cmd - y
        Y = dy
        mask = np.abs(X) > 1e-3
        tau_inv_est = np.linalg.lstsq(X[mask][:, None], Y[mask], rcond=None)[0][0]
        tau = np.clip(1.0 / tau_inv_est, 1e-5, 1.0)
        return tau

    def simulate_response(y0, y_cmd, tau):
        y_sim = np.zeros_like(y_cmd)
        y_sim[0] = y0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            y_sim[i] = y_sim[i - 1] + (y_cmd[i - 1] - y_sim[i - 1]) * dt / tau
        return y_sim

    # --- p-axis
    y_p = data["p_filt"]
    y_p_cmd = data["p_cmd"]
    tau_p = estimate_tau(y_p, y_p_cmd, "p")
    y_p_sim = simulate_response(y_p[0], y_p_cmd, tau_p)

    # --- q-axis
    y_q = data["q_filt"]
    y_q_cmd = data["q_cmd"]
    tau_q = estimate_tau(y_q, y_q_cmd, "q")
    y_q_sim = simulate_response(y_q[0], y_q_cmd, tau_q)

    # --- r-axis
    y_r = data["r_filt"]
    y_r_cmd = data["r_cmd"]
    tau_r = estimate_tau(y_r, y_r_cmd, "r")
    y_r_sim = simulate_response(y_r[0], y_r_cmd, tau_r)

    # --- T_norm (from -az_filt)
    y_T = -data["az_filt"]  # flip sign: positive thrust = positive acc
    y_T_cmd = data["T_cmd"]
    tau_T = estimate_tau(y_T, y_T_cmd, "T_norm")
    y_T_sim = simulate_response(y_T[0], y_T_cmd, tau_T)

    # Store all
    taus = {
        "tau_p": tau_p,
        "tau_q": tau_q,
        "tau_r": tau_r,
        "tau_T": tau_T,
    }

    # --- Plotting
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axs[0].plot(t, y_p_cmd, "--k", label="p_cmd")
    axs[0].plot(t, y_p, label="p_measured")
    axs[0].plot(t, y_p_sim, label="p_simulated")
    axs[0].set_title(f"p (τ = {tau_p:.3f} s)")

    axs[1].plot(t, y_q_cmd, "--k", label="q_cmd")
    axs[1].plot(t, y_q, label="q_measured")
    axs[1].plot(t, y_q_sim, label="q_simulated")
    axs[1].set_title(f"q (τ = {tau_q:.3f} s)")

    axs[2].plot(t, y_r_cmd, "--k", label="r_cmd")
    axs[2].plot(t, y_r, label="r_measured")
    axs[2].plot(t, y_r_sim, label="r_simulated")
    axs[2].set_title(f"r (τ = {tau_r:.3f} s)")

    axs[3].plot(t, y_T_cmd, "--k", label="T_cmd")
    axs[3].plot(t, y_T, label="-az_filt (thrust)")
    axs[3].plot(t, y_T_sim, label="T_simulated")
    axs[3].set_title(f"T_norm (τ = {tau_T:.3f} s)")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend()

    axs[-1].set_xlabel("time [s]")
    fig.tight_layout()
    plt.show()

    if PRINT_CI:
        tau_p, ci_p, tauinv_p, _, _ = tau_and_ci_from_channel(
            data["p_filt"], data["p_cmd"], data["t"]
        )
        tau_q, ci_q, tauinv_q, _, _ = tau_and_ci_from_channel(
            data["q_filt"], data["q_cmd"], data["t"]
        )
        tau_r, ci_r, tauinv_r, _, _ = tau_and_ci_from_channel(
            data["r_filt"], data["r_cmd"], data["t"]
        )
        tau_T, ci_T, tauinv_T, _, _ = tau_and_ci_from_channel(
            -data["az_filt"], data["T_cmd"], data["t"]
        )
        p_pct = ci_halfwidth_pct(tau_p, ci_p[0], ci_p[1])
        q_pct = ci_halfwidth_pct(tau_q, ci_q[0], ci_q[1])
        r_pct = ci_halfwidth_pct(tau_r, ci_r[0], ci_r[1])
        T_pct = ci_halfwidth_pct(tau_T, ci_T[0], ci_T[1])
        print(
            f"taus (95% CI):  p={tau_p:.4f} [{ci_p[0]:.4f}, {ci_p[1]:.4f}] (±{p_pct:.2f}% hw)   "
            f"q={tau_q:.4f} [{ci_q[0]:.4f}, {ci_q[1]:.4f}] (±{q_pct:.2f}% hw)   "
            f"r={tau_r:.4f} [{ci_r[0]:.4f}, {ci_r[1]:.4f}] (±{r_pct:.2f}% hw)   "
            f"T={tau_T:.4f} [{ci_T[0]:.4f}, {ci_T[1]:.4f}] (±{T_pct:.2f}% hw)"
        )

    return taus


data = load_paparazzi_log(
    # "/Users/merlijnbroekers/Downloads/20250626-101027.csv",
    # "/Users/merlijnbroekers/Downloads/frpn_log/20250821-135009.csv",
    "/Users/merlijnbroekers/Downloads/no_smooth_no_rate/part_1/20251007-111624.csv",
    # "/Users/merlijnbroekers/Downloads/reward_smoothing/motor/gamma_10_dr10/20250916-124753.csv",
    rpm_min=2500,
    rpm_max=12000,
)

# Use the PSD to determine where to set your filtering frequencies --> noise removal
plot_psd(data, fmax=512)

filter_flight_data(
    data, cutoff_acc=20, cutoff_rates=8, cutoff_att=200, cutoff_motor=250
)

# # # # # # 1) Thrust & planar drag
k_w, k_x_motor, k_y_motor = fit_thrust_drag_model(data)

# # # # # # # 2) Actuator (rpm dynamics) Note the filtered omega is not used here as there is little noise
if PRINT_CI:
    params_motor = fit_actuator_model_with_ci(data)
else:
    params_motor = fit_actuator_model(data)

# # # # 3) Moments (roll/pitch/yaw)
moments = fit_moments_model(data)

k_x, k_y = fit_ctbr_planar_drag(data, v_thresh=0.1)

taus = estimate_time_constants_ctbr(data)
