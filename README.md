# Drone Interception

Reinforcement Learning-Based Guidance and Control for Aerial-to-Aerial Pest Interception

## Abstract

---

## Overview

This repository contains the codebase for a reinforcement learning-based drone-pest interception controller. The codebase supports simulation training and a pipe-line for real-world deployment.

INCLUDE .GIF HERE LATER !!!!

## Repository Structure

```
Drone_Interception/
├── src/                    # Core simulation and models
│   ├── models/             # Drone and pest/evaders
│   ├── simulation/         # Simulation environment (RL training & evaluation)
│   ├── control_laws/       # Control law interface
│   └── utils/              # Configuration, logging, utilities
├── scripts/
│   ├── training/           # Training scripts
│   ├── analysis/           # Analysis and evaluation tools
│   └── system_identification/  # System ID tool
└── deployment/             # Real-world deployment code
    ├── ivy_senders/        # Stream evader trajectories to Paparazzi (Ivy)
    └── extract_validate.py # Model extraction and validation
```

**Note**: Training datasets and trained models are not included in the repository but may be available upon request.

## Key Components

### Core Simulation (`src/`)

- **Models**: 
  - **Pursuers**: Motor-level, CTBR-INDI, and Acceleration-level controllers
  - **Evaders**: Moth trajectories, Pliska trajectories, reactive RL evaders, and classic evaders (see [`src/models/evaders/README.md`](src/models/evaders/README.md) for CSV format requirements)
- **Simulation**: 
  - **RL Training Environment**: Vectorized Gymnasium environment for training (see [`src/simulation/README.md`](src/simulation/README.md))
  - **Evaluation Simulation**: Standalone simulation for independent model evaluation
- **Control Laws**: API for control laws
- **Utils**: Configuration management, observation builders, reward functions, logging (see [`src/utils/README.md`](src/utils/README.md) for detailed config parameter documentation)

### Training (`scripts/training/`)

The training module provides tools for training reinforcement learning agents using PPO (Proximal Policy Optimization). See [`scripts/training/README.md`](scripts/training/README.md) for detailed documentation.


### Analysis (`scripts/analysis/`)

Analysis tools for evaluating trained models comparing different configurations in simulation, and processing physical drone flights (CyberZoo/PATS). See [`scripts/analysis/README.md`](scripts/analysis/README.md) for detailed documentation.

### Deployment (`deployment/`)

Methology for deploying trained models on physical drones and scripts for streaming target infornmation to paparazzi. See [`deployment/README.md`](deployment/README.md) for detailed documentation.

### System Identification (`scripts/system_identification/`)

Tools for identifying drone model parameters from paparazzi flight logs.

## Quick Start

### Installation

To ensure compatibility, use python 3.10. Nessecary for compatibility with Stable-Baselines3 2.X and TensorFlow 1.x

```bash
pip install -r requirements.txt
```

### Basic Usage

#### Run a Simulation

```bash
python scripts/main.py
```

This runs a single simulation using the configuration in `src/utils/config.py`, yielding an animation of the drone-pest pursuit. 

#### Train a Model

To train a single drone controller against an evader using the configuration in `src/utils/config.py`

Example:
```bash
python scripts/training/single_env.py
```

## Configuration

The configuration file is `src/utils/config.py`. See [`src/utils/README.md`](src/utils/README.md) for detailed documentation of all configuration parameters.


## Datasets

Training datasets used in this work may be available upon request. For information on creating compatible evader trajectory datasets, see [`src/models/evaders/README.md`](src/models/evaders/README.md).
