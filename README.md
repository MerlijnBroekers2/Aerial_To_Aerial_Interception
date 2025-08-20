# RL Motor Pursuer Policies for Interception
This repository trains **reinforcement learning (RL)** policies that drive a **motor‑level quadrotor pursuer** to intercept a moving evader. It provides a vectorized simulation environment, physics‑based motor/airframe model, and evader generators (replayed from logs). Configuration lives in `config.py` and controls dynamics, observations, rewards, and domain randomization.

## Repository layout (key files)
- `training.py` – PPO training entrypoint (Stable‑Baselines3) with vectorized environment and callbacks.
- `simulation.py` – rollout for evaluation/visualization.
- `env_pursuit_evasion.py` – `PursuitVecEnv` (Gymnasium‑style) that wires pursuer and evader, builds observations, computes rewards/dones.
- `vec_pursuer_Motor.py` / `pursuer_Motor.py` – physics models for the pursuer at **motor level** (vectorized and single‑env variants).
- `vec_moth_evader.py` / `vec_pliska_evader.py` – evaders that replay trajectories from logs with optional filtering/noise.
- `config.py` – scenario/configuration; see **Configuration** below.

## Observations & Actions (summary)

- **Actions**: for the motor‑level pursuer the action is 4‑D in `[-1, 1]` (one per motor). Actions are filtered by first‑order motor time constants and mapped to motor speeds; thrust and body torques are produced by the provided polynomial motor model.

- **Observations**: the environment builds a concatenated vector including (normalized) pursuer/evader positions and velocities, line‑of‑sight (LOS) unit vectors and rates, and optional extras (see `src.utils.observation`). 

## Rewards & Termination (summary)

- **Capture / interception**: episode terminates when the distance between pursuer and evader is ≤ `CAPTURE_RADIUS`. Set `STOP_ON_INTERCEPTION=True` to end early.
- **Time limit**: episode also ends at `TOTAL_TIME` seconds.
- **Reward**: configurable via `REWARD_TYPE` and weights (see `get_reward` in `src.utils.reward`). 

## Frames & Gravity

Everything in this repo is expressed in **NED** (North‑East‑Down) unless stated otherwise. 
Evaders replayed from logs can be converted (e.g., ENU→NED) inside the evader classes—check their constructors if your data is ENU.

## Configuration (`config.py`)
`config.py` contains a single `CONFIG` dictionary. The most important sections are:

- **Global**: `DT`, `TOTAL_TIME`, `CAPTURE_RADIUS`, `ENV_BOUND`, etc.
- **EVADER**: source of trajectories, filtering/noise, start/end sample indices or times.
- **PURSUER**: dynamics and limits. For the motor model: thrust/torque polynomials, motor speed limits, actuator time constants, drag, etc
