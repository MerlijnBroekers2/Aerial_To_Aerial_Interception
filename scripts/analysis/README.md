# Analysis Module

This directory contains tools for analyzing trained models and evaluating performance in both simulation and real-world scenarios.

## Overview

The analysis module provides:
- **Simulation-based evaluation**: Test models in simulation with various metrics
- **Real-world data processing**: Analyze flight data from physical drones
- **Visualization**: Generate plots, animations, and summaries

## Directory Structure

```
analysis/
├── cyberzoo_analysis/         # Analysis of real flights (CyberZoo)
│   ├── collect_real_metrics_from_dirs.py
│   ├── metrics_multiple_trials_real_life.py
│   ├── reward_smoothing_metric_collection.py
│   └── animation_real.py
├── pats_analysis/             # Analysis of PATS flight data
│   ├── batch_metric_inbounds_computation.py
│   ├── metric_inbounds_computation.py
│   ├── pats_verification.py
│   ├── boundary_plane_generation.py
│   └── check_point_in_plane.py
├── utils/                     # Shared analysis utilities
│   ├── metrics_utils.py
│   ├── plot_utils.py
│   ├── stats_utils.py
│   ├── run_utils.py
│   └── notebook_helpers.py
├── *.ipynb                    # Jupyter notebooks for interactive analysis
└── feature_weight.py          # Analyze feature importance in policies
```

## Analysis Workflows

### Simulation Analysis

Use Jupyter notebooks for interactive simulation-based analysis:

- **`abstraction_level_analysis.ipynb`**: Compare performance across abstraction levels
- **`domain_randomization_analysis.ipynb`**: Analyze impact of domain randomization
- **`reward_smoothing_analysis.ipynb`**: Evaluate reward smoothing strategies
- **`observation_space_analysis.ipynb`**: Compare different observation configurations
- **`action_history_analysis.ipynb`**: Analyze impact of action history

### Real-World Analysis: CyberZoo

**Directory**: `cyberzoo_analysis/`

Analyze data from physical drone flights conducted at the CyberZoo facility.

#### Collect Metrics from Directories

**Script**: `collect_real_metrics_from_dirs.py`

Processes flight logs and computes metrics:

```bash
python scripts/analysis/cyberzoo_analysis/collect_real_metrics_from_dirs.py
```

**Functionality**:
- Scans directories for flight log CSV files
- Extracts metrics (interceptions, closest distances, etc.)
- Aggregates across multiple trials
- Generates summary statistics
- Saves results to CSV for further analysis

**Output**: `analysis_out/cyberzoo/real_metrics_*.csv`

#### Multiple Trials Analysis

**Script**: `metrics_multiple_trials_real_life.py`

Analyzes multi-trial flight data:
- Handles trials with spoof segments (multiple evader runs per flight)
- Computes per-trial and aggregate statistics

#### Reward Smoothing Metrics

**Script**: `reward_smoothing_metric_collection.py`

Specialized analysis for reward smoothing experiments:
- Compares different smoothing gamma values
- Generates comparison plots
- Statistical summaries

#### Real Flight Animation

**Script**: `animation_real.py`

Creates 3D animations from real flight data:
- Visualizes pursuer and evader trajectories
- Shows interception events
- Rotating camera view
- Can save as GIF or video

### Real-World Analysis: PATS

**Directory**: `pats_analysis/`

Analyzes data from PATS flights.

#### Metric Computation

**Script**: `metric_inbounds_computation.py`

Computes in-bounds metrics for single flight logs:
- Checks trajectory against boundary planes
- Identifies interception events
- Derives performance metrics

**Script**: `batch_metric_inbounds_computation.py`

Batch processing version for multiple flights:
- Processes entire directories of flight logs
- Aggregates metrics across trials
- Generates comparison plots (boxplots, etc.)
- Supports filtering by controller type, subset, etc.

#### Boundary Plane Generation

**Script**: `boundary_plane_generation.py`

Generates boundary plane definitions for geofencing:
- Takes PATS frame FOV and depth camera details
- Defines planes in NED coordinates
- Used for checking trajectory compliance

#### Verification Tools

**Script**: `pats_verification.py`

Verification and sanity checks for PATS data processing.

**Script**: `check_point_in_plane.py`

Utility for checking if points lie within defined planes.

### Feature Weight Analysis

**Script**: `feature_weight.py`

Analyzes feature importance in trained policies:
- Extracts first-layer weights from neural network
- Computes feature importance scores
- Visualizes which observations are most important
- Helps understand what the agent is focusing on
