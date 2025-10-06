"""
Defines and creates the TVM tensor operators for evaluation.
"""

import numpy as np
from tvm import te, topi

def create_conv2d_operator(config: dict):
    """
    Creates a TVM conv2d operator and random input data based on config dict.

    Args:
        config (dict): A dictionary containing the opearator parameters.

    Returns:
        tuple: A tuple containing (data_np, kernel_np, data_placeholder, kernel_placeholder, conv_output).
    """

    batch_size = config["batch_size"]
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    in_height, in_width = config["in_height"], config["in_width"]
    kernel_height, kernel_width = config["kernel_height"], config["kernel_width"]
    strides, padding, dtype = config["strides"], config["padding"], config["dtype"]

    # Define shapes
    data_shape = (batch_size, in_channels, in_height, in_width)
    kernel_shape = (out_channels, in_channels, kernel_height, kernel_width)

    # Generate random input data
    data_np = np.random.uniform(-1, 1, data_shape).astype(dtype)
    kernel_np = np.random.uniform(-1, 1, kernel_shape).astype(dtype)

    # Define TVM placeholders
    data = te.placeholder(data_shape, name="data", dtype=dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=dtype)

    # Define the conv2d operation using TOPI
    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype=dtype)

    return data_np, kernel_np, data, kernel, conv
