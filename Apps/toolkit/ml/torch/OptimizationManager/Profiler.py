import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
import plotly.express as px
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
from abc import ABC, abstractmethod
import torch.utils.tensorboard as tb
import json

class ProfilerStrategy(ABC):
    @abstractmethod
    def profile(self, module: nn.Module, input_data: Any) -> Any:
        pass

class PyTorchProfilerStrategy(ProfilerStrategy):
    def __init__(self, use_cuda: bool = False, record_shapes: bool = False, profile_memory: bool = False,
                 with_stack: bool = False, with_flops: bool = False):
        self.use_cuda = use_cuda
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops

    def profile(self, module: nn.Module, input_data: Any):
        with profiler.profile(use_cuda=self.use_cuda, record_shapes=self.record_shapes,
                              profile_memory=self.profile_memory, with_stack=self.with_stack,
                              with_flops=self.with_flops) as prof:
            module(input_data)
        return prof

class HTAProfilerStrategy(ProfilerStrategy):
    def profile(self, module: nn.Module, input_data: Any) -> Any:
        # Implement HTA-specific profiling logic here
        pass

class Profiler:
    def __init__(self, strategy: ProfilerStrategy = PyTorchProfilerStrategy(), config: Optional[Dict[str, Any]] = None) -> None:
        self.strategy = strategy
        self.last_profile_result = None
        self.tb_writer = None
        self.config = config or {}
        self.cache = {}

    def set_strategy(self, strategy: ProfilerStrategy) -> None:
        self.strategy = strategy

    def profile_module(self, module: nn.Module, input_data: Any, use_tensorboard: bool = False,
                       log_dir: str = './log', force_profile: bool = False,
                       schedule: Optional[torch.profiler.schedule] = None) -> None:
        """
        Profile a PyTorch module using the current profiling strategy.

        Args:
            module (nn.Module): The PyTorch module to profile.
            input_data (Any): Input data for the module.
            use_tensorboard (bool): Whether to use TensorBoard for visualization.
            log_dir (str): Directory to save TensorBoard logs.
            force_profile (bool): Whether to force profiling even if cached results exist.
            schedule (Optional[torch.profiler.schedule]): Schedule for long-running jobs.
        """
        cache_key = f"{module.__class__.__name__}_{hash(tuple(input_data))}"
        if not force_profile and cache_key in self.cache:
            self.last_profile_result = self.cache[cache_key]
            return

        # Warm-up CUDA to ensure accurate performance benchmarking
        if torch.cuda.is_available():
            module(input_data)

        if use_tensorboard:
            self.tb_writer = tb.SummaryWriter(log_dir)
            profiler_kwargs = {
                "activities": [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                "schedule": schedule or torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                "on_trace_ready": torch.profiler.tensorboard_trace_handler(log_dir),
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": True,
                "with_flops": True
            }
            with torch.profiler.profile(**profiler_kwargs) as prof:
                module(input_data)
            self.last_profile_result = prof
        else:
            self.last_profile_result = self.strategy.profile(module, input_data)

        self.cache[cache_key] = self.last_profile_result

    def analyze_profile(self, sort_by: str = 'self_cpu_time_total', row_limit: int = 10) -> str:
        """
        Analyze the last profiling result and return a formatted table.

        Args:
            sort_by (str): Column to sort the results by.
            row_limit (int): Maximum number of rows to display.

        Returns:
            str: Formatted table of profiling results.
        """
        if self.last_profile_result is None:
            return "No profiling data available. Run profile_module first."

        if isinstance(self.last_profile_result, profiler.ProfilerResult):
            return self.last_profile_result.key_averages().table(
                sort_by=sort_by, row_limit=row_limit
            )
        else:
            # Implement HTA-specific analysis here
            pass

    def suggest_optimizations(self) -> List[str]:
        """
        Suggest optimizations based on the last profiling result.

        Returns:
            List[str]: A list of optimization suggestions.
        """
        if self.last_profile_result is None:
            return ["No profiling data available. Run profile_module first."]

        suggestions = []

        if isinstance(self.last_profile_result, profiler.ProfilerResult):
            averages = self.last_profile_result.key_averages()

            for event in averages:
                if event.key == "aten::copy_" and event.cpu_time_total > 1e-3:
                    suggestions.append(
                        f"High copy time detected ({event.cpu_time_total:.2f}s). "
                        "Consider keeping tensors on the same device to avoid unnecessary copies."
                    )
                
                if event.cpu_memory_usage > 1e6:  # More than 1 MB
                    suggestions.append(
                        f"High memory usage detected in {event.key} ({event.cpu_memory_usage / 1e6:.2f} MB). "
                        "Consider using lower precision (e.g., float instead of double) or optimizing tensor operations."
                    )

                if event.cuda_time_total > 1e-3 and event.cuda_time_total > 10 * event.cpu_time_total:
                    suggestions.append(
                        f"High CUDA time detected in {event.key} ({event.cuda_time_total:.2f}s). "
                        "Consider optimizing GPU operations or using more efficient CUDA kernels."
                    )

            # Check for GPU utilization
            gpu_util = self.last_profile_result.total_average().cuda_time_total / self.last_profile_result.total_average().cpu_time_total
            if gpu_util < 0.5:
                suggestions.append(
                    f"Low GPU utilization detected ({gpu_util:.2f}). "
                    "Consider increasing batch size or parallelizing more operations on GPU."
                )

        else:
            # Implement HTA-specific optimization suggestions here
            pass

        if not suggestions:
            suggestions.append("No specific optimizations suggested based on the current profile.")

        return suggestions

    def get_temporal_breakdown(self, visualize: bool = False) -> pd.DataFrame:
        """
        Generate a temporal breakdown of GPU usage.

        Args:
            visualize (bool): Whether to generate a visualization of the breakdown.

        Returns:
            pd.DataFrame: A dataframe containing the temporal breakdown.
        """
        # Implement temporal breakdown logic here
        # This is a placeholder implementation
        data = {
            'Rank': [0, 1, 2, 3],
            'Idle Time': [10, 15, 12, 8],
            'Compute Time': [60, 55, 58, 62],
            'Non-compute Time': [30, 30, 30, 30]
        }
        df = pd.DataFrame(data)

        if visualize:
            fig = go.Figure(data=[
                go.Bar(name='Idle Time', x=df['Rank'], y=df['Idle Time']),
                go.Bar(name='Compute Time', x=df['Rank'], y=df['Compute Time']),
                go.Bar(name='Non-compute Time', x=df['Rank'], y=df['Non-compute Time'])
            ])
            fig.update_layout(barmode='stack', title='Temporal Breakdown by Rank')
            fig.show()

        return df

    def get_idle_time_breakdown(self, ranks: List[int] = [0], consecutive_kernel_delay: float = 30e-9,
                                show_idle_interval_stats: bool = False, visualize_pctg: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate an idle time breakdown.

        Args:
            ranks (List[int]): List of ranks to analyze.
            consecutive_kernel_delay (float): Threshold for kernel wait time.
            show_idle_interval_stats (bool): Whether to show idle interval statistics.
            visualize_pctg (bool): Whether to visualize percentages or absolute times.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two dataframes containing idle time breakdown and statistics.
        """
        # Implement idle time breakdown logic here
        # This is a placeholder implementation
        idle_time_df = pd.DataFrame({
            'Rank': ranks,
            'Stream': ['Default'] * len(ranks),
            'Host Wait': [50, 45, 55],
            'Kernel Wait': [10, 15, 5],
            'Other Wait': [40, 40, 40]
        })

        idle_time_stats_df = pd.DataFrame({
            'Rank': ranks,
            'Stream': ['Default'] * len(ranks),
            'Count': [100, 95, 105],
            'Min': [0.1, 0.2, 0.1],
            'Max': [10, 9, 11],
            'Mean': [5, 4.5, 5.5],
            'Std': [2, 1.8, 2.2]
        })

        if visualize_pctg:
            fig = go.Figure(data=[
                go.Bar(name='Host Wait', x=idle_time_df['Rank'], y=idle_time_df['Host Wait']),
                go.Bar(name='Kernel Wait', x=idle_time_df['Rank'], y=idle_time_df['Kernel Wait']),
                go.Bar(name='Other Wait', x=idle_time_df['Rank'], y=idle_time_df['Other Wait'])
            ])
            fig.update_layout(barmode='stack', title='Idle Time Breakdown by Rank (%)')
            fig.show()

        return idle_time_df, idle_time_stats_df

    def get_memory_stats(self) -> pd.DataFrame:
        """
        Generate memory statistics from the profiling result.

        Returns:
            pd.DataFrame: A dataframe containing memory statistics.
        """
        if self.last_profile_result is None:
            return pd.DataFrame()

        memory_stats = []
        for event in self.last_profile_result.key_averages():
            if event.cpu_memory_usage > 0 or event.cuda_memory_usage > 0:
                memory_stats.append({
                    'Event': event.key,
                    'CPU Memory (MB)': event.cpu_memory_usage / 1e6,
                    'CUDA Memory (MB)': event.cuda_memory_usage / 1e6,
                    'CPU Time (ms)': event.cpu_time_total * 1000,
                    'CUDA Time (ms)': event.cuda_time_total * 1000
                })

        return pd.DataFrame(memory_stats)

    def visualize_memory_usage(self):
        """
        Visualize memory usage over time using Plotly.
        """
        if self.last_profile_result is None:
            print("No profiling data available. Run profile_module first.")
            return

        memory_events = []
        for event in self.last_profile_result.events():
            if event.cpu_memory_usage > 0 or event.cuda_memory_usage > 0:
                memory_events.append({
                    'ts': event.time_range.start,
                    'CPU Memory (MB)': event.cpu_memory_usage / 1e6,
                    'CUDA Memory (MB)': event.cuda_memory_usage / 1e6
                })

        df = pd.DataFrame(memory_events)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ts'], y=df['CPU Memory (MB)'], mode='lines', name='CPU Memory'))
        fig.add_trace(go.Scatter(x=df['ts'], y=df['CUDA Memory (MB)'], mode='lines', name='CUDA Memory'))
        fig.update_layout(title='Memory Usage Over Time', xaxis_title='Timestamp', yaxis_title='Memory (MB)')
        fig.show()

    def export_results(self, format: str = 'json', filename: str = 'profile_results') -> None:
        """
        Export profiling results to a file.

        Args:
            format (str): The format to export ('json', 'csv', or 'chrome_trace').
            filename (str): The name of the file to save (without extension).
        """
        if self.last_profile_result is None:
            raise ValueError("No profiling data available. Run profile_module first.")

        if format == 'json':
            with open(f"{filename}.json", 'w') as f:
                json.dump(self.last_profile_result.key_averages().table(row_limit=-1), f)
        elif format == 'csv':
            pd.DataFrame(self.last_profile_result.key_averages().table()).to_csv(f"{filename}.csv", index=False)
        elif format == 'chrome_trace':
            self.last_profile_result.export_chrome_trace(f"{filename}.json")
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def compare_profiles(self, other_profile: 'Profiler') -> pd.DataFrame:
        """
        Compare this profile with another profile.

        Args:
            other_profile (Profiler): Another Profiler instance to compare with.

        Returns:
            pd.DataFrame: A dataframe containing the comparison of the two profiles.
        """
        if self.last_profile_result is None or other_profile.last_profile_result is None:
            raise ValueError("Both profiles must have data available.")

        self_data = pd.DataFrame(self.last_profile_result.key_averages().table())
        other_data = pd.DataFrame(other_profile.last_profile_result.key_averages().table())

        merged_data = pd.merge(self_data, other_data, on='Name', suffixes=('_self', '_other'))
        
        # Calculate differences and percentage changes
        for col in self_data.columns:
            if col != 'Name' and col in other_data.columns:
                merged_data[f'{col}_diff'] = merged_data[f'{col}_self'] - merged_data[f'{col}_other']
                merged_data[f'{col}_pct_change'] = ((merged_data[f'{col}_self'] - merged_data[f'{col}_other']) / merged_data[f'{col}_other']) * 100

        return merged_data

    def analyze_stack_traces(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Analyze stack traces from the profiling result.

        Args:
            top_n (int): Number of top stack traces to return.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing stack trace information.
        """
        if self.last_profile_result is None:
            raise ValueError("No profiling data available. Run profile_module first.")

        stack_traces = []
        for event in self.last_profile_result.key_averages():
            if event.stack:
                stack_traces.append({
                    'event': event.key,
                    'cpu_time': event.cpu_time_total,
                    'cuda_time': event.cuda_time_total,
                    'stack': event.stack_repr
                })

        # Sort by CPU time (you can change this to sort by CUDA time if needed)
        stack_traces.sort(key=lambda x: x['cpu_time'], reverse=True)

        return stack_traces[:top_n]

    def visualize_operator_distribution(self):
        """
        Visualize the distribution of operator types using a pie chart.
        """
        if self.last_profile_result is None:
            print("No profiling data available. Run profile_module first.")
            return

        operator_counts = {}
        for event in self.last_profile_result.key_averages():
            op_type = event.key.split('::')[0]
            operator_counts[op_type] = operator_counts.get(op_type, 0) + 1

        labels = list(operator_counts.keys())
        values = list(operator_counts.values())

        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title='Distribution of Operator Types')
        fig.show()

    def analyze_bottlenecks(self, threshold_ms: float = 10.0) -> List[Dict[str, Any]]:
        """
        Analyze potential bottlenecks in the profiled code.

        Args:
            threshold_ms (float): Time threshold in milliseconds to consider an operation as a bottleneck.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing bottleneck information.
        """
        if self.last_profile_result is None:
            raise ValueError("No profiling data available. Run profile_module first.")

        bottlenecks = []
        for event in self.last_profile_result.key_averages():
            cpu_time_ms = event.cpu_time_total * 1000
            cuda_time_ms = event.cuda_time_total * 1000

            if cpu_time_ms > threshold_ms or cuda_time_ms > threshold_ms:
                bottlenecks.append({
                    'event': event.key,
                    'cpu_time_ms': cpu_time_ms,
                    'cuda_time_ms': cuda_time_ms,
                    'input_shapes': event.input_shapes if hasattr(event, 'input_shapes') else None,
                    'stack_trace': event.stack_repr if hasattr(event, 'stack_repr') else None
                })

        return sorted(bottlenecks, key=lambda x: max(x['cpu_time_ms'], x['cuda_time_ms']), reverse=True)

    def visualize_execution_timeline(self):
        """
        Visualize the execution timeline of events using a Gantt chart.
        """
        if self.last_profile_result is None:
            print("No profiling data available. Run profile_module first.")
            return

        events = []
        for event in self.last_profile_result.events():
            events.append({
                'Task': event.name,
                'Start': event.time_range.start,
                'Finish': event.time_range.end,
                'Resource': 'CPU' if event.device_type == 0 else 'CUDA'
            })

        df = pd.DataFrame(events)

        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource")
        fig.update_layout(title='Execution Timeline', xaxis_title='Time (μs)', yaxis_title='Operations')
        fig.show()

    def get_flop_estimate(self) -> float:
        """
        Estimate the number of floating-point operations (FLOPs) performed during profiling.

        Returns:
            float: Estimated number of FLOPs.
        """
        if self.last_profile_result is None:
            raise ValueError("No profiling data available. Run profile_module first.")

        total_flops = 0
        for event in self.last_profile_result.key_averages():
            if hasattr(event, 'flops'):
                total_flops += event.flops

        return total_flops
