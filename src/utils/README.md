# Utilities Module

This module contains configuration management, observation building, reward computation, logging, and other shared utilities.

## Configuration Parameters

The main configuration is defined in `config.py`. Below is a detailed explanation of all parameters:

### Global Settings

#### `LOG_LEVEL`
- **Type**: String
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
- **Default**: `"INFO"`
- **Description**: Controls verbosity of logging throughout the system.

#### `DT`
- **Description**: Simulation time step in seconds. This is the fundamental time discretization for all dynamics integration.

#### `TOTAL_TIME`
- **Description**: Maximum simulation duration for **evaluation** simulations. Episodes will run for this duration unless terminated early.

#### `TIME_LIMIT`
- **Description**: Maximum number of steps for **RL training** episodes. Actual time = `TIME_LIMIT * DT`.

#### `STOP_ON_INTERCEPTION`
- **Description**: Whether to terminate simulation immediately upon interception. If `False`, simulation continues and may record multiple interceptions.

#### `INTERCEPTION_RADIUS`
- **Description**: Distance threshold for detecting interceptions in **evaluation** simulations.

#### `CAPTURE_RADIUS`
- **Description**: Distance threshold for detecting captures in **RL training** (terminates episode).

---

### Observation Configuration (`OBSERVATIONS`)

#### `OBS_MODE`
- **Type**: String
- **Options**: 
  - `"rel_pos"`: Relative position only (direction + magnitude)
  - `"rel_pos_body"`: Relative position in body frame
  - `"rel_pos+vel"`: Relative position and velocity
  - `"rel_pos+vel_body"`: Relative position and velocity in body frame (recommended)
  - `"pos+vel"`: Absolute positions and velocities
  - `"all"`: All available observations
  - `"all_body_no_phi_rate"`: All observations except phi rate
  - `"rel_pos_vel_los_rate"`: Relative pos/vel + line-of-sight rate
  - `"rel_pos_vel_los_rate_body"`: Above in body frame
- **Description**: Core observation mode defining what information the agent receives.
- The `observations.py` module handles building observation vectors from state information.


#### `INCLUDE_HISTORY`
- **Description**: Whether to include historical observations in the observation vector.

#### `HISTORY_STEPS`
- **Description**: Number of previous observations to include when `INCLUDE_HISTORY=True`. Each step adds another copy of the base observation.

#### `INCLUDE_ACTION_HISTORY`
- **Description**: Whether to include previous actions in the observation vector.

#### `ACTION_HISTORY_STEPS`
- **Description**: Number of previous actions to include when `INCLUDE_ACTION_HISTORY=True`.

#### `MAX_HISTORY_STEPS`
- **Description**: Must be greater than `HISTORY_STEPS` and `ACTION_HISTORY_STEPS`. Pre-allocates buffer size for history.

#### `ACTION_DIM`
- **Description**: Action space dimensionality. Must match abstraction level:
  - Motor: `4`
  - CTBR: `4`
  - Acceleration: `3`

#### `OPTIONAL_FEATURES`
- **Options**:
  - `"attitude"`: Euler angles (3D)
  - `"attitude_mat"`: First two columns of rotation matrix (6D)
  - `"rates"`: Angular rates p, q, r (3D)
  - `"T_force"`: Thrust force (1D)
  - `"omega_norm"`: Motor speeds (4D)
- **Description**: Additional features appended to base observation for improved stability.

---

### Reward Configuration

#### `reward_type`
- **Options**:
  - `"effective_gain"`: Reward based on distance reduction relative to evader movement
  - `"distance_gain"`: Simple distance reduction reward
  - `"simple"`: Distance-based reward with capture bonus
  - `"reinier"`: Scaled distance shaping from Reinier's thesis
- **Modifiers** (append with `+`):
  - `+smooth`: Enables smoothing penalty (uses `SMOOTHING_GAMMA`)
  - `+no_rate`: Disables angular rate penalty. Active by default

#### `SMOOTHING_GAMMA`
- **Description**: Smoothing strength when using `+smooth` modifier. Higher values penalize control oscillations more. Typically ranges from 0.0 to 50.0.

#### `RATE_PENALTY`
- **Description**: Penalty coefficient for angular rates. Applied as `-RATE_PENALTY * ||rates||` unless `+no_rate` modifier is used.

#### `OUT_OF_BOUNDS_PENALTY`
- **Description**: Termination penalty when pursuer goes out of bounds. Applied as negative reward at episode end.

#### `CAPTURE_PENALTY`
- **Description**: Reward/penalty for successful capture.

---

### Pursuer Configuration (`PURSUER`)

#### `MODEL`
- **Options**: `"motor"`, `"ctbr_indi"`, `"acc_indi"`
- **Description**: Control abstraction level

#### `INITIAL_POS`, `INITIAL_VEL`, etc.
- **Description**: Initial conditions for pursuer (position, velocity, attitude, rates, motor speeds).

#### `INIT_RADIUS`
- **Description**: Randomization radius for initial position (spherical).

#### `MAX_ACCELERATION`
- **Description**: Maximum acceleration for scaling acceleration commands.

#### `MAX_SPEED`
- **Description**: Maximum speed (relevant only for first-order acceleration model).

#### `ACTUATOR_TAU`
- **Description**: Actuator time constant for only first-order acceleration model.

#### `POSITION_NOISE_STD`, `VELOCITY_NOISE_STD`
- **Description**: Standard deviation of noise added to position/velocity measurements of the drone in simulation

#### `BUTTER_ACC_FILTER_CUTOFF_HZ`
- **Description**: Low-pass filter cutoff frequency for acceleration commands.

#### `BOUNDARIES`
Nested configuration for geofencing:

##### `ENV_BOUNDS`
- **Description**: Fallback axis-aligned bounds (used if `PLANES` is not defined).

##### `BOUNDARY_MARGIN`
- **Description**: Distance from boundary where penalty starts.

##### `BOUNDARY_PENALTY_WEIGHT`
- **Description**: Global multiplier for boundary penalties.

##### `BOUNDARY_MODE`
- **Options**: `"sum"`, `"max"`
- **Description**: How to combine penalties from multiple boundary planes:
  - `"sum"`: Sum all penalties (corner regions get higher penalty)
  - `"max"`: Use maximum penalty (corner regions don't accumulate)

##### `PLANES`
- **Description**: Defines geofence as half-planes. Point is inside if `dot(n, p - p0) >= 0` for all planes.

#### `actuator_time_constants`
- **Keys**: `"p"`, `"q"`, `"r"`, `"T"`
- **Description**: Time constants for actuator dynamics (seconds).

#### `actuator_limits`
- **Keys**: `"p"`, `"q"`, `"r"`, `"T"`, `"bank_angle"`
- **Description**: Hard limits on control commands.

#### `domain_randomization_pct`
- **Description**: Domain randomization percentages (0.0 to 1.0) for various parameters:

#### `CONTROLLER`
Configuration for pursuer controller:

##### `type`
- **Options**: `"rl"`, `"frpn"`
- **Description**: Controller type. FRPN only avaliable for acceleration abstraction level

##### `policy_path`
- **Description**: Path to trained RL model (`.zip` file) when `type="rl"`.

##### `params`
- **Description**: Parameters for classical FRPN controller.

---

### Evader Configuration (`EVADER`)

#### `MODEL`
- **Options**: `"moth"`, `"pliska"`, `"rl"`, `"classic"`
- **Description**: Evader type.

#### Moth-Specific Parameters

##### `CSV_FILE`
- **Description**: Path to CSV file with moth trajectory data.

##### `NOISE_STD`
- **Default**: `0.0`
- **Description**: Standard deviation of noise added to positions.

##### `FILTER_TYPE`
- **Options**: `"passthrough"`, `"ekf"`
- **Description**: Filtering method for trajectory data.

##### `FILTER_PARAMS`
- **Keys**:
  - `"q_acc"`: Process noise (for EKF)
  - `"r_pos"`: Measurement noise (for EKF)
  - `"pos_window_samples"`: Window size for windowed filter
  - `"vel_window_samples"`: Window size for velocity
  - `"vel_from_filtered_pos"`: Whether to derive velocity from filtered positions

#### Pliska-Specific Parameters

##### `PLISKA_CSV_FOLDER`
- **Description**: Folder containing Pliska CSV files.

##### `PLISKA_POSITION_BOUND`
- **Description**: Pre-scale position bound for trajectories.

##### `PLISKA_SPEED_MULT`
- **Default**: `1.0`
- **Description**: Speed multiplier for trajectories.

##### `PLISKA_VEL_FROM_POS`
- **Description**: Whether to derive velocity from position differentiation.

##### `EVAL_USE_FILTERED_AS_GT`
- **Description**: Whether to use filtered positions as ground truth (affects observation/reward computation).

#### RL Evader Parameters

##### `RL_MODEL_PATH`
- **Description**: Path to trained RL evader model.

##### `MAX_ACCEL`, `MAX_SPEED`
- **Description**: Maximum acceleration and speed for RL evader.

##### `INIT_POS`, `INIT_VEL`
- **Description**: Initial conditions for RL evader.

##### `BOUNDARIES`
- **Description**: Geofence constraints for RL evader

#### Classic Evader Parameters

##### `PATH_TYPE`
- **Options**: `"figure_eight"`, etc.
- **Description**: Predefined path type.

##### `RADIUS`
- **Description**: radius parameter for circular/figure-8 trajectories.

##### `VELOCITY_MAGNITUDE`
- **Description**: Constant velocity magnitude.

---

### Policy Configuration

#### `POLICY_KWARGS`
- **Description**: Neural network architecture:
  - `"pi"`: Actor network layers
  - `"vf"`: Critic network layers

---

### Dataset Paths

#### `MOTH_FOLDER`
- **Description**: Folder containing moth CSV trajectory files (used for training).
**Note**: Training datasets used in this work may be available upon request.





