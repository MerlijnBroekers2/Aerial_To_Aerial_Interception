# Ivy Senders

This directory contains scripts for streaming evader trajectories to the Paparazzi autopilot via the Ivy bus. These scripts allow real-world testing of trained pursuer agents by sending target (evader) position and velocity data to the drone's flight controller.

## Overview

All senders follow a similar pattern:
1. **Load evader trajectory data** (from CSV files or RL models)
2. **Connect to Ivy bus** (Paparazzi's message bus)
3. **Subscribe to drone state** (position, velocity from INS/GPS)
4. **Send evader state** as `TARGET_INFO` messages at configured rate
5. **Monitor/Display realtime pursuit/evasion** (optional)

## Available Senders

### 1. `moth_data_sender.py`

**Purpose**: Streams multiple moth trajectories in a loop with automatic geofence planning and spawn management.

**Features**:
- Loads multiple moth CSV files from a folder
- Automatically scales and offsets trajectories to fit within geofence bounds
- Plans spawn positions to avoid clustering
- Switches to next evader on interception or timeout
- Includes real-time 3D visualization (optional)

**Usage**:
```bash
python deployment/ivy_senders/moth_data_sender.py
```

**Modes**:
- `"stream"`: Stream to Ivy bus with real-time plot
- `"preview"`: Preview trajectory planning without streaming

**Output**: Sends `TARGET_INFO` messages with evader position/velocity in NED frame.

### 2. `moth_data_single_sender.py`

**Purpose**: Streams a single moth trajectory for multi-trial testing with spoofing support.

**Features**:
- Single trajectory per run (specified in CONFIG)
- Multiple trials with optional spoofing (holds target at fixed point before each trial)
- Configurable trial duration and spoof duration
- Useful for repeated testing 

**Usage**:
```bash
python deployment/ivy_senders/moth_data_single_sender.py
```

### 3. `reactive_sender.py`

**Purpose**: Streams trajectories from a trained RL evader agent (reactive evader).

**Features**:
- Loads trained PPO evader model (set by defining RL_MODEL_PATH)
- Generates reactive trajectories in real-time based on pursuer state
- Evader responds to pursuer position/velocity
- Supports boundary constraints
- Multi-trial support with spoofing

**Usage**:
```bash
python deployment/ivy_senders/reactive_sender.py
```

**How it works**:
1. Loads RL evader policy (PPO model)
2. Subscribes to drone INS messages to get pursuer state
3. At each timestep:
   - Builds observation from relative state
   - Queries evader policy for action
   - Integrates action to get next position/velocity
   - Sends to Ivy bus

**Note**: This reactive evader needs pursuer state, so both must be on the same Ivy bus.

---

### 4. `pliska_sender.py`

**Purpose**: Streams Pliska trajectory data retrieved from 10.1109/LRA.2024.3451768.

**Features**:
- Loads Pliska CSV files from folder
- Similar geofence planning as moth sender
- Supports EKF filtering of trajectories
- 3D visualization with rotating camera (optional)

**Usage**:
```bash
python deployment/ivy_senders/pliska_sender.py
```

## Common Components

### Ivy Bridge

All senders use an `IvyBridge` class (or similar) that:
- Connects to Ivy bus using `IvyMessagesInterface`
- Subscribes to `INS` messages to get drone position (NED, from INS fixed-point)
- Sends `TARGET_INFO` messages with evader state

**Message Format**:
```python
msg = PprzMessage("datalink", "TARGET_INFO")
msg["enu_x"], msg["enu_y"], msg["enu_z"] = pos_ned  # ! NED coordinates EVEN THOUGH PPRZ-MESSAGE SAYS ENU
msg["enu_xd"], msg["enu_yd"], msg["enu_zd"] = vel_ned  # ! NED velocities EVEN THOUGH PPRZ-MESSAGE SAYS ENU
msg["ac_id"] = AC_ID
ivy.send(msg)
```

**Note**: Fields are named `enu_*` but contain NED coordinates (Paparazzi convention used).

### Troubleshooting
- Check Ivy bus address is correct
- Ensure Paparazzi is running
- Verify network connectivity
- Check `AC_ID` matches drone's aircraft ID

### Integration with Pursuer

The pursuer (drone) must be configured to:
1. Subscribe to `TARGET_INFO` messages
2. Parse evader position/velocity from `enu_*` fields (as NED)


