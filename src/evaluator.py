"""
Handles the compilation and performance evaluation of a given TVM schedule.
"""
import tvm
import numpy as np

class PerformanceEvaluator:
    def __init__(self, eval_config: dict):
        self.target = eval_config["target"]
        self.device = getattr(tvm, eval_config["device"])(eval_config["device_id"])
        self.eval_config = eval_config

    def evaluate(
        self,
        schedule: tvm.te.Schedule,
        tensors_to_build: list,
        input_data: tuple
    ) -> float:
        """
        Builds the function, runs it, and measures its execution time.

        Args:
            schedule (tvm.te.Schedule): The scheduled operator.
            tensors_to_build (list): List of input and output tensors for tvm.build.
            input_data (tuple): A tuple of numpy arrays for the inputs (e.g., data, kernel).

        Returns:
            float: The mean execution time in seconds.
        """
        # Build the function
        func = tvm.build(schedule, tensors_to_build, target=self.target)

        # Prepare TVM NDArrays on the target device
        tvm_inputs = [tvm.nd.array(data, device=self.device) for data in input_data]
        
        # Prepare output buffer
        output_shape = tensors_to_build[-1].shape
        output_dtype = tensors_to_build[-1].dtype
        output_tvm = tvm.nd.empty(output_shape, output_dtype, self.device)

        # Create time evaluator
        evaluator = func.time_evaluator(
            func.entry_name,
            self.device,
            number=self.eval_config["number"],
            repeat=self.eval_config["repeat"]
        )

        # Measure performance
        mean_time = evaluator(*tvm_inputs, output_tvm).mean
        return mean_time
