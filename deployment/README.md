# Deployment

This directory contains tools for deploying trained models on physical drones and running real-world tests.

## Overview

The deployment module provides:
1. **Model conversion**: Export PyTorch models to C for embedded systems
2. **Validation**: Verify model correctness after conversion
3. **Ivy senders**: Stream evader trajectories to Paparazzi autopilot for testing

## Model Deployment

### Extracting Models to C

**File**: `extract_validate.py`

This script converts trained PPO models to C headers for embedded deployment:

1. **Extracts weights** from PyTorch model state dict
2. **Generates C header** (`ppo_weights.h`) with weight arrays
3. **Compiles shared library** (`libppo_controller.so` or `.dylib`)
4. **Validates** by comparing PyTorch and C outputs

**Usage**:
```bash
python deployment/extract_validate.py \
    --model path/to/trained/model.zip \
    --csrc deployment/ppo_controller.c
```

Note, ensure that the input and output dimensions in the extract validate script match that of `ppo_controller.c`.

**Configuration**:
```python
HEADER_NAME = "ppo_weights.h"
LIB_BASE = "libppo_controller"
TOLERANCE = 1e-5              # Validation tolerance
NUM_TESTS = 500               # Number of validation tests
OBS_RANGE = (-100.0, 100.0)   # Observation range for testing
INPUT_DIM = 18                # Observation dimension 
OUTPUT_DIM = 3                # Action dimension
```

**Output**:
- `ppo_weights.h`: C header with network weights
- `libppo_controller.{so,dylib}`: Compiled shared library

**Validation**:
The script runs random test observations through both PyTorch and C implementations and checks outputs match within tolerance. Exit code 0 = success, 1 = failure.

### C Controller Template

**File**: `ppo_controller.c`

Template C implementation of a PPO policy forward pass. You'll need to:
1. Copy this file to your build directory
2. Include the generated `ppo_weights.h`
3. Implement the forward pass using the weight arrays
4. Compile into a shared library

The template assumes:
- ReLU activations
- Linear layers
- Tanh output activation (for bounded actions)

**Key Functions**:
- `ppo_forward()`: Main forward pass function
- Input: observation array
- Output: action array

### Using Ivy Senders

See [`ivy_senders/README.md`](ivy_senders/README.md) for detailed documentation on streaming evader trajectories.


