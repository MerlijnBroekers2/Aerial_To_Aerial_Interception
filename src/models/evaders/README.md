# Evader Models

This directory contains implementations of different evader (target) types used for training and evaluation.

## Evader Types

- **Moth Evader**: Replays moth trajectories from CSV files
- **Pliska Evader**: Replays Pliska trajectory data
- **Reactive RL Evader**: RL-trained evader that responds to pursuer state
- **Classic Evader**: Predefined paths (e.g., figure-eight)

## Moth Evader CSV Format

The `MothEvader` class loads trajectories from CSV files. To use your own dataset, CSV files must adhere to the following structure:

### Required Columns

The CSV must contain the following columns (exact names, case-sensitive):

#### Position Columns (PATS frame)
- `sposX_insect`: X position
- `sposY_insect`: Y position  
- `sposZ_insect`: Z position

#### Velocity Columns (PATS frame)
- `svelX_insect`: X velocity
- `svelY_insect`: Y velocity
- `svelZ_insect`: Z velocity

#### Time Column
- `elapsed` (or `time`, `timestamp`): Elapsed time in seconds (will be normalized to start at 0)

### Coordinate Frame

**Important**: The CSV file should contain positions and velocities in **PATS** frame:
- **X (Left)**: Left of PATS camera frame 
- **Y (Up)**: Above camera horizon positive
- **Z (Back)**: Behind the camera

The `MothEvader` automatically converts to **NED (North-East-Down)** frame internally using the transformation:
```python
NED = [-Z, -X, -Y]  # from PATS
```

### CSV Format Requirements
- **Delimiter**: Semicolon (`;`) - this is the default, but the reader is flexible
- **Headers**: First row must contain column names
- **Data Types**: Position, velocity, and time columns must be numeric (float)

### Example CSV Structure

```csv
elapsed;sposX_insect;sposY_insect;sposZ_insect;svelX_insect;svelY_insect;svelZ_insect
0.000;1.234;2.345;0.567;0.123;0.456;-0.789
0.010;1.245;2.356;0.578;0.125;0.458;-0.790
0.020;1.256;2.367;0.589;0.127;0.460;-0.791
...
```

### Trajectory Processing

The `MothEvader` performs the following processing:

1. **Load CSV**: Reads position, velocity, and time columns
2. **Frame Conversion**: Converts from PATS/ENU to NED
3. **Time Normalization**: Shifts time to start at 0
4. **Noise Addition** (optional): Adds Gaussian noise to positions if `NOISE_STD > 0`
5. **Filtering** (optional): Applies filter (passthrough or EKF) to smooth trajectories
6. **Velocity Computation**: Either uses CSV velocities or derives from filtered positions


## Training Datasets

The training datasets used in this work are available upon request

