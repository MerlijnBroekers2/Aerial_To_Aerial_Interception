# Simulation Module

This directory contains the simulation environments used for both **reinforcement learning training** and **independent evaluation** of trained models.

## Overview

The simulation module provides two distinct types of environments:

1. **RL Training Environments** (`env_pursuit_evasion.py`): Vectorized Gymnasium environments for training agents
2. **Evaluation Simulation** (`simulation.py`): Standalone simulation for independent model evaluation

## RL Training Environment

**File**: `env_pursuit_evasion.py`

**Purpose**: Provides a vectorized reinforcement learning environment compatible with Stable-Baselines3.

## Independent Evaluation Simulation

**File**: `simulation.py`

**Purpose**: Standalone simulation for evaluating trained models independently of RL training framework.

**Key Features**:
- **Single episode**: Runs one simulation at a time
- **Detailed history**: Records full state trajectory
- **Interception tracking**: Records all interception events
- **No RL framework**: Direct pursuer/evader interaction
- **Flexible control**: Supports both RL and classical controllers

