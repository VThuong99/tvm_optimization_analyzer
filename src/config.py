"""
Configuration for the TVM operator and evaluation settings.
"""

# Defines parameters for a typical 3x3 conv2d layer from ResNet-50 conv3_x. See: https://arxiv.org/abs/1512.03385
CONV2D_CONFIG = {
    "batch_size": 1,
    "in_channels": 128,
    "out_channels": 128,
    "in_height": 28,
    "in_width": 28,
    "kernel_height": 3,
    "kernel_width": 3,
    "strides": (1, 1),
    "padding": (1, 1),
    "dtype": "float32"
}

# Defines parameters for the performance measurement.
EVALUATION_CONFIG = {
    "target": "llvm",  # Target for compilation (e.g., "llvm" for CPU, "cuda" for GPU)
    "device": "cpu",   # Device to run on
    "device_id": 0,
    "number": 100,     # Number of runs within a single repeat
    "repeat": 10       # Number of times to repeat the measurement
}


# Parameters used by specific optimization strategies.
OPTIMIZATION_PARAMS = {
    "vector_width": 8, # Use for AVX2/AVX512 support on the CPU
    "tile_oc": 8,
    "tile_ow": 8,
    "tile_ic": 4
}