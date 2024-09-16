from loguru import logger
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.benchmark import Timer, Compare, Fuzzer, FuzzedTensor, FuzzedParameter, CallgrindStats
from typing import Callable, List, Dict, Any, Optional, Union
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import wraps

class Benchmarker:
    def __init__(self, default_num_runs: int = 100, default_num_threads: int = 1,
                 default_min_run_time: float = 0.1):
        self.results = []
        self.default_num_runs = default_num_runs
        self.default_num_threads = default_num_threads
        self.default_min_run_time = default_min_run_time

    def _log_error(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper

    @_log_error
    def benchmark_function(self, func: Callable, input_generator: Callable, 
                           label: str, num_runs: Optional[int] = None, 
                           num_threads: Optional[int] = None,
                           min_run_time: Optional[float] = None) -> Timer.TimerResult:
        """
        Benchmark a function with given inputs.

        Args:
            func (Callable): The function to benchmark.
            input_generator (Callable): A function that generates inputs for the benchmarked function.
            label (str): A label for this benchmark.
            num_runs (Optional[int]): Number of runs for the benchmark. Defaults to self.default_num_runs.
            num_threads (Optional[int]): Number of threads to use. Defaults to self.default_num_threads.
            min_run_time (Optional[float]): Minimum run time for the benchmark. Defaults to self.default_min_run_time.

        Returns:
            Timer.TimerResult: The measurement result.
        """
        inputs = input_generator()
        num_runs = num_runs or self.default_num_runs
        num_threads = num_threads or self.default_num_threads
        min_run_time = min_run_time or self.default_min_run_time

        timer = Timer(
            stmt='func(*inputs)',
            globals={'func': func, 'inputs': inputs},
            num_threads=num_threads,
            label=label
        )
        measurement = timer.blocked_autorange(min_run_time=min_run_time)
        self.results.append(measurement)
        logger.info(f"Benchmarked {label}: {measurement}")
        return measurement

    @_log_error
    def compare_results(self) -> Compare:
        """Compare all benchmarked results."""
        if not self.results:
            raise ValueError("No benchmark results to compare.")
        compare = Compare(self.results)
        compare.trim_significant_figures()
        compare.colorize()
        return compare

    def print_comparison(self) -> None:
        """Print the comparison of all benchmarked results."""
        compare = self.compare_results()
        compare.print()

    @_log_error
    def save_results(self, filename: str) -> None:
        """Save benchmark results to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        logger.info(f"Results saved to {filename}")

    @_log_error
    def load_results(self, filename: str) -> None:
        """Load benchmark results from a file."""
        with open(filename, 'rb') as f:
            self.results = pickle.load(f)
        logger.info(f"Results loaded from {filename}")

    def generate_fuzzed_inputs(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Generate fuzzed inputs for benchmarking."""
        fuzzer = Fuzzer(
            parameters=[
                FuzzedParameter('k0', minval=1, maxval=10000, distribution='loguniform'),
                FuzzedParameter('k1', minval=1, maxval=10000, distribution='loguniform'),
            ],
            tensors=[
                FuzzedTensor('x', size=('k0', 'k1'), min_elements=128, max_elements=10000000, 
                             probability_contiguous=0.6)
            ],
            seed=0,
        )
        return list(fuzzer.take(num_samples))

    @_log_error
    def benchmark_with_fuzzed_inputs(self, func: Callable, num_samples: int = 10) -> None:
        """Benchmark a function with fuzzed inputs."""
        for tensors, tensor_params, params in self.generate_fuzzed_inputs(num_samples):
            sub_label = f"{params['k0']:<6} x {params['k1']:<4} {'(discontiguous)' if not tensor_params['x']['is_contiguous'] else ''}"
            measurement = Timer(
                stmt='func(x)',
                setup='',
                globals={'func': func, 'x': tensors['x']},
                label='Fuzzed input benchmark',
                sub_label=sub_label,
            ).blocked_autorange(min_run_time=0.1)
            self.results.append(measurement)
            logger.info(f"Benchmarked with fuzzed input: {sub_label}")

    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self.results = []
        logger.info("All benchmark results cleared")

    @_log_error
    def collect_callgrind(self, func: Callable, input_generator: Callable, 
                          label: str) -> CallgrindStats:
        """
        Collect Callgrind stats for a function.

        Args:
            func (Callable): The function to analyze.
            input_generator (Callable): A function that generates inputs for the analyzed function.
            label (str): A label for this analysis.

        Returns:
            CallgrindStats: The collected Callgrind statistics.
        """
        inputs = input_generator()
        timer = Timer(
            stmt='func(*inputs)',
            globals={'func': func, 'inputs': inputs},
            label=label
        )
        stats = timer.collect_callgrind()
        logger.info(f"Collected Callgrind stats for {label}")
        return stats

    @_log_error
    def visualize_results(self, filename: Optional[str] = None) -> None:
        """
        Visualize benchmark results.

        Args:
            filename (Optional[str]): If provided, save the plot to this file.
        """
        if not self.results:
            raise ValueError("No benchmark results to visualize.")

        data = []
        for result in self.results:
            data.append({
                'Label': result.label,
                'Median (us)': result.median * 1e6,
                'IQR': result.iqr * 1e6
            })

        df = pd.DataFrame(data)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Label', y='Median (us)', data=df)
        plt.errorbar(x=range(len(df)), y=df['Median (us)'], 
                     yerr=df['IQR']/2, fmt='none', c='black', capsize=5)
        plt.title('Benchmark Results')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            logger.info(f"Visualization saved to {filename}")
        else:
            plt.show()

    def calculate_speedup(self, baseline_label: str) -> Dict[str, float]:
        """
        Calculate speedup relative to a baseline.

        Args:
            baseline_label (str): The label of the baseline benchmark.

        Returns:
            Dict[str, float]: A dictionary of labels and their speedups.
        """
        baseline = next((r for r in self.results if r.label == baseline_label), None)
        if not baseline:
            raise ValueError(f"No benchmark found with label '{baseline_label}'")

        speedups = {}
        for result in self.results:
            if result.label != baseline_label:
                speedup = baseline.median / result.median
                speedups[result.label] = speedup
                logger.info(f"Speedup of {result.label} relative to {baseline_label}: {speedup:.2f}x")

        return speedups

# Example usage
if __name__ == "__main__":
    benchmarker = Benchmarker()

    def example_function(x: torch.Tensor) -> torch.Tensor:
        return x.mul(2)

    def input_generator() -> List[torch.Tensor]:
        return [torch.randn(1000, 1000)]

    benchmarker.benchmark_function(example_function, input_generator, "Example benchmark")
    benchmarker.benchmark_with_fuzzed_inputs(example_function, num_samples=5)
    benchmarker.print_comparison()
    benchmarker.visualize_results("benchmark_results.png")
    speedups = benchmarker.calculate_speedup("Example benchmark")
    print("Speedups:", speedups)

    callgrind_stats = benchmarker.collect_callgrind(example_function, input_generator, "Example Callgrind")
    print(callgrind_stats)