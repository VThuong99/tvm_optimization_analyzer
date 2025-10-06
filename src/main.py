import tvm
from tvm import te

from config import CONV2D_CONFIG, EVALUATION_CONFIG, OPTIMIZATION_PARAMS
from operator_factory import create_conv2d_operator
from evaluator import PerformanceEvaluator
from visualizer import plot_performance_results
from optimizer import methods

def main():
    # Create the base operator and input data
    data_np, kernel_np, data, kernel, conv = create_conv2d_operator(CONV2D_CONFIG)
    tensors_to_build = [data, kernel, conv]

    # Initialize the performance evaluator
    evaluator = PerformanceEvaluator(EVALUATION_CONFIG)

    # Define the list of strategies
    strategies_to_test = [
        methods.BaselineStrategy(OPTIMIZATION_PARAMS),
        methods.ParallelStrategy(OPTIMIZATION_PARAMS),
        methods.TilingStrategy(OPTIMIZATION_PARAMS),
        methods.VectorizeStrategy(OPTIMIZATION_PARAMS),
        # methods.TilingAndParallelStrategy(OPTIMIZATION_PARAMS),
        # methods.TilingAndVectorizeStrategy(OPTIMIZATION_PARAMS),
        methods.AllCombinedStrategy(OPTIMIZATION_PARAMS),
    ]

    # Run experiments and collect results
    results = {}
    for i, strategy in enumerate(strategies_to_test):
        strategy_name = str(strategy)
        print(f"{i+1}. Evaluating Strategy: {strategy_name}")

        # Create a fresh schedule for each strategy
        schedule = te.create_schedule(conv.op)

        # Apply the optimization strategy
        strategy.apply(schedule, conv)

        # Evaluate the performance
        mean_time = evaluator.evaluate(
            schedule,
            tensors_to_build,
            (data_np, kernel_np)
        )
        results[strategy_name] = mean_time

        print(f"   -> Measured Latency: {mean_time * 1000:.3f} ms")
        print("-" * 60)

    plot_performance_results(results)

if __name__ == "__main__":
    main()
