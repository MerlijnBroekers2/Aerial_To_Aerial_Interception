# Training Module

This directory contains scripts for training reinforcement learning agents to intercept evading targets.

## Overview

The training module uses **Proximal Policy Optimization (PPO)** from Stable-Baselines3 to train pursuer agents. Training is organized around configuration sweeps that systematically explore different hyperparameters and design choices.

## Directory Structure

```
training/
├── sweeps/                        # Configuration sweep scripts
│   ├── abstraction_level_training.py
│   ├── domain_randomisation_training.py
│   ├── reward_smoothing_training.py
│   ├── observation_space_training.py
│   ├── observation_history_training.py
│   └── action_history_training.py
├── utils/
│   └── training_utils.py          # Shared training utilities
├── reactive_evader_training.py    # Train reactive RL evaders
├── self_play.py                   # Self-play training
└── single_env.py                  # Single environment training script
```

## Train 

To train under the current configuration run

```bash
python scripts/training/single_env.py
```

## Training Sweeps

### Abstraction Level Training

**File**: `sweeps/abstraction_level_training.py`

Trains agents at different control abstraction levels:
- **Motor**: Direct motor commands 
- **CTBR**: Collective thrust and body rates commands 
- **Acceleration**: Inertial acceleration commands


### Domain Randomization Training

**File**: `sweeps/domain_randomisation_training.py`

Systematically explores domain randomization parameters to improve sim-to-real transfer:
- Trains agents with varying levels of domain randomization (0%, 10%, 20%, 30%, etc.)

**Usage**:
```bash
python scripts/training/sweeps/domain_randomisation_training.py
```

### Reward Smoothing Training

**File**: `sweeps/reward_smoothing_training.py`

Experiments with reward smoothing to reduce control oscillations:

- Tests different smoothing gamma values (0.0, 1.0, 2.0, 5.0, etc.)
- Combines with rate penalty options

### Observation Space Training

**File**: `sweeps/observation_space_training.py`

Tests different observation space configurations:

### Observation History Training

**File**: `sweeps/observation_history_training.py`

Investigates the impact of including historical observations:
- Varies history length (0, 1, 2, 4, 8 steps, etc.)


### Action History Training

**File**: `sweeps/action_history_training.py`

Tests including previous actions in the observation:

- Varies action history length (1, 2, 4, 8, 16 steps)
- Helps agents learn smoother control policies
- Particularly useful for reducing control oscillations

## Monitoring Training

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir trained_models/<experiment_name>/<config_tag>/logs
```

