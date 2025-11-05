CONFIG = {
    "LOG_LEVEL": "INFO",
    "DT": 1 / 100,
    "TOTAL_TIME": 12,  # evaluation-simulation clock
    "TIME_LIMIT": 1200,  # Reinforcement learning max steps done
    "STOP_ON_INTERCEPTION": False,
    "INTERCEPTION_RADIUS": 0.15,  # Evaluation config
    "CAPTURE_RADIUS": 0.15,  # RL-learning config
    "POLICY_KWARGS": dict(
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64])
    ),  # actor/critic MLP structure
    "OBSERVATIONS": {
        "OBS_MODE": "rel_pos+vel_body",  # core observation mode
        "MAX_HISTORY_STEPS": 1,  # this MUST be > steps when using obs/act history
        "INCLUDE_HISTORY": False,
        "HISTORY_STEPS": 0,  # amount of histroical observations to include
        "INCLUDE_ACTION_HISTORY": False,
        "ACTION_HISTORY_STEPS": 0,  # amount of histroical actions to include
        "ACTION_DIM": 4,  # this must be set to match the abstraction level [acc=3, ctbr=4, motor=4]
        "OPTIONAL_FEATURES": [  # additional observations that may be added for stability
            # "attitude",
            "attitude_mat",
            "rates",
            # "T_force",
            "omega_norm",
        ],
    },
    "reward_type": "effective_gain",
    # Modifiers (append to reward_type):
    # +smooth   -> subtract bounded smoothing penalty using gamma_s
    # +no_rate  -> disable subtraction of gamma_r * ||rates||
    "SMOOTHING_GAMMA": 0.0,  # sets the value of gamma_s
    "RATE_PENALTY": 0.001,  # Reward smoothing penalty depending on the angular rate
    "OUT_OF_BOUNDS_PENALTY": 5,  # Termination penalty given when the pursuer goes out of bounds
    "CAPTURE_PENALTY": -2,  # When not training with boundary adherence, you may opt to set this higher for faster learing
    "MOTH_FOLDER": "/Users/merlijnbroekers/Desktop/Drone_Interception/evader_datasets/opogona_old/top_moths",  # dataset used for learning
    "PURSUER": {
        "MODEL": "motor",  # abstraction leve
        "LOG_LEVEL": "DEBUG",
        "INITIAL_POS": [1.0, 0.0, 1.0],
        "INIT_RADIUS": 0.25,
        "INITIAL_VEL": [0.0, 0.0, 0.0],
        "INITIAL_ATTITUDE": [0.0, 0.0, 0.0],
        "INITIAL_RATES": [0.0, 0.0, 0.0],
        "INITIAL_OMEGA": [0.0, 0.0, 0.0],
        "MAX_ACCELERATION": 18,  # Maximum acceleration used for scaling acceleration commands
        "MAX_SPEED": 20,  # relevant only for 1st order acceleration model
        "ACTUATOR_TAU": 0.05,  # relevant only for 1st order acceleration model
        "POSITION_NOISE_STD": 0.0,
        "VELOCITY_NOISE_STD": 0.0,
        "BUTTER_ACC_FILTER_CUTOFF_HZ": 8,  # low-pass filtering of acceleration commands
        "gravity": 9.81,
        "BOUNDARIES": {
            # fall back boundaries if no planes are defined
            "ENV_BOUNDS": {"x": (-10.0, 10.0), "y": (-10.0, 10.0), "z": (-10.0, 10)},
            "BOUNDARY_MARGIN": 0.1,  # distance from wall where penalty starts
            "BOUNDARY_PENALTY_WEIGHT": 0.25,  # global multiplier
            "BOUNDARY_MODE": "sum",  # "sum" (default) or "max" (no corner boost)
            "PLANES": [  # Set to none to fallback to ENV_bounds
                {
                    "n": [0.5961325493620492, 0.7253743710122877, 0.3441772878468769],
                    "p0": [0.0, 0.0, 0.0],
                },
                {
                    "n": [0.5961325493620492, -0.7253743710122877, 0.3441772878468769],
                    "p0": [0.0, 0.0, 0.0],
                },
                {
                    "n": [0.8571673007021123, 0.0, -0.5150380749100543],
                    "p0": [0.0, 0.0, 0.0],
                },
                {
                    "n": [-0.017452406437283352, 0.0, 0.9998476951563913],
                    "p0": [0.0, 0.0, 0.0],
                },
                {
                    "n": [0.8660254037844387, 0.0, 0.49999999999999994],
                    "p0": [0.47631397208144133, 0.0, 0.27499999999999997],
                },
                {
                    "n": [-0.8660254037844387, -0.0, -0.49999999999999994],
                    "p0": [3.464101615137755, 0.0, 1.9999999999999998],
                },
                {"n": [0.0, 0.0, -1.0], "p0": [0.0, 0.0, 2.0]},
                {"n": [0.0, 0.0, 1.0], "p0": [0.0, 0.0, -0.1]},
                {"n": [-1.0, 0.0, 0.0], "p0": [3.0, 0.0, 0.0]},
                {"n": [1.0, 0.0, 0.0], "p0": [0.55, 0.0, 0.0]},
            ],
        },
        "actuator_time_constants": {
            "p": 1.65e-01,
            "q": 1.51e-01,
            "r": 4.44e-01,
            "T": 8.6e-02,
            # "phi": 0.6,
            # "theta": 0.6,
        },
        "actuator_limits": {
            "p": (-3.0, 3.0),
            "q": (-3.0, 3.0),
            "r": (-2.0, 2.0),
            "T": (1.41, 18.4),
            "bank_angle": 20.0,
        },
        "delta_a_limits": {
            "min": [-6.0, -6.0, -9.0],
            "max": [6.0, 6.0, 9.0],
        },
        "attitude_pd": {
            "kp": {
                "phi": 20.0,
                "theta": 20.0,
                "psi": 20.0,
            },
            "kd": {
                "phi": 1.0,
                "theta": 1.0,
                "psi": 1.0,
            },
        },
        "motor": {
            "k_w": -3.70e-06,
            "k_x": -1.02e-04,
            "k_y": -1.05e-04,
            "k_p1": 1.10e-04,
            "k_p2": -1.04e-04,
            "k_p3": -1.03e-04,
            "k_p4": 1.06e-04,
            "k_q1": 8.93e-05,
            "k_q2": 8.70e-05,
            "k_q3": -8.29e-05,
            "k_q4": -8.95e-05,
            "k_r1": 7.21e-03,
            "k_r2": -7.44e-03,
            "k_r3": 1.07e-02,
            "k_r4": -7.80e-03,
            "k_r5": 1.56e-03,
            "k_r6": -1.39e-03,
            "k_r7": 1.50e-03,
            "k_r8": -1.31e-03,
            "w_min": 2.9942e02,
            "w_max": 1.14127e03,
            "curve_k": 1.0e00,
            "tau": 6.0e-02,
        },
        "drag": {"kx_acc_ctbr": 3.10e-01, "ky_acc_ctbr": 3.28e-01},
        "domain_randomization_pct": {
            k: 0.00
            for k in [
                "g",
                "kx_acc_ctbr",
                "ky_acc_ctbr",
                "taup",
                "tauq",
                "taur",
                "tauT",
                "tauphi",
                "tau_actuator",
                "tautheta",
                "max_accel",
                "max_speed",
                "p_lo",
                "p_hi",
                "q_lo",
                "q_hi",
                "r_lo",
                "r_hi",
                "T_lo",
                "T_hi",
                "bank_angle",
                "delta_a_min",
                "delta_a_max",
                "k_x",
                "k_y",
                "k_w",
                "k_p1",
                "k_p2",
                "k_p3",
                "k_p4",
                "k_q1",
                "k_q2",
                "k_q3",
                "k_q4",
                "k_r1",
                "k_r2",
                "k_r3",
                "k_r4",
                "k_rd1",
                "k_rd2",
                "k_rd3",
                "k_rd4",
                "w_min",
                "w_max",
                "tau",
                "curve_k",
            ]
        }
        | {  # dict merge
            "init_radius": 0.0,
        },
        "CONTROLLER": {
            "type": "rl",  #   "rl" | "frpn"
            "policy_path": "/Users/merlijnbroekers/Desktop/Drone_Interception/trained_models/abstraction_level_testing/motor/models/best_model.zip",
            "params": {  # for FRPN these are used only
                "lambda_": 180,
                "pp_weight": 0.25,
                "max_acceleration": 18,
            },
        },
    },
    "EVADER": {
        "MODEL": "moth",  # "classic", "moth", "rl", "pliska"
        # Pliska evaders only
        # ----------------------------------------------------------------------------------------------------
        "EVAL_USE_FILTERED_AS_GT": True,  # False = Meas/GT observation/reward split PRIVALEDGED REWARD, True = Meas/Meas split
        "PLISKA_VEL_FROM_POS": True,  # wether velocity is derived from position through differentiation or loaded from csv
        "PLISKA_POSITION_BOUND": 3.0,
        "PLISKA_SPEED_MULT": 1.0,
        "PLISKA_CSV_FOLDER": "/Users/merlijnbroekers/Desktop/Drone_Interception/evader_datasets/pliska_csv",
        "NOISE_STD": 0.00,  # noise added to positions only
        # ----------------------------------------------------------------------------------------------------
        "CSV_FILE": "evader_datasets/opogona_old/top_moths/log_itrk32.csv",  # single moth/pliska trajectory
        # Classic evaders only
        # ----------------------------------------------------------------------------------------------------
        "PATH_TYPE": "figure_eight",
        "RADIUS": 1.0,
        "VELOCITY_MAGNITUDE": 2.0,
        "FILTER_TYPE": "passthrough",  # moth has [passthrough, ekf (legacy/untested)], pliska has [passthrough, windowed]
        # ----------------------------------------------------------------------------------------------------
        "FILTER_PARAMS": {
            "process_noise": 1e-4,  # ekf q_acc
            "measurement_noise": 1e-2,  # ekf r_pos
            "pos_window_samples": 10,
            "vel_window_samples": 5,
            # ! Similar to PLISKA_VEL_FROM_POS but this triggers wether the velocity should be derived from filtered positions instead of raw
            "vel_from_filtered_pos": True,
        },
        # RL evaders only
        # ----------------------------------------------------------------------------------------------------
        "RL_MODEL_PATH": "final_models_selfplay/ALT_0911_1111/round_4_evader/evader_final.zip",
        "MAX_ACCEL": 50,  # max acceleration of the RL evader
        "MAX_SPEED": 2.0,  # max velocity of the RL evader
        "INIT_POS": [0, 0, 0],
        "INIT_VEL": [0, 0, 0],
        "BOUNDARIES": {
            "ENV_BOUNDS": {
                "x": (-10.25, 10.25),
                "y": (-10.25, 10.25),
                "z": (-10.25, 10.25),
            },
            "BOUNDARY_MARGIN": 0.5,  # distance from wall where penalty starts
            "BOUNDARY_PENALTY_WEIGHT": 0.5,  # global multiplier (kept)
            "BOUNDARY_MODE": "sum",  # "sum" (default) or "max" (no corner boost)
        },
        # ----------------------------------------------------------------------------------------------------
    },
}
