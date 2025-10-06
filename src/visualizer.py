import matplotlib.pyplot as plt

def plot_performance_results(results: dict):
    if not results or "Baseline" not in results:
        print("Error: Baseline result is missing or results are empty. Cannot generate plot.")
        return

    method_names = list(results.keys())

    latencies = [results[method] * 1000 for method in method_names]  # in ms
    speedups = [results["Baseline"] / results[method] for method in method_names]

    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Bar chart for time (ms)
    color = 'tab:blue'
    ax1.set_ylabel('Time (ms)', color=color, fontsize=14)
    bars = ax1.bar(method_names, latencies, color=color, width=0.6)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.set_xticklabels(method_names, rotation=30, ha="right", fontsize=12)

    # Add data labels on bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.01, f'{yval:.3f}', ha='center', va='bottom')

    # Create a second y-axis for relative speedup
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Relative speedup', color=color, fontsize=14)
    ax2.plot(method_names, speedups, color=color, marker='o', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.grid(axis='y', linestyle=':', color='gray')
    ax2.axhline(y=1.0, color='gray', linestyle=':', label='Baseline Speedup')

    for i, txt in enumerate(speedups):
        ax2.annotate(f'{txt:.2f}x', (method_names[i], speedups[i]), textcoords="offset points", xytext=(0,10), ha='center', color=color)

    fig.tight_layout()
    plt.title('Performance Comparison of TVM Optimization Strategies', fontsize=16)
    plt.show()