import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from loguru import logger
from tqdm import tqdm
import traceback
import os
import logging
from typing import Optional, Callable

class Compiler:
    def __init__(self):
        self.compile_enabled = True
        self.compile_mode = "reduce-overhead"
        self.fullgraph = False
        
        # Cache-related attributes
        self.fx_graph_cache_enabled = self._get_env_bool('TORCHINDUCTOR_FX_GRAPH_CACHE', True)
        self.fx_graph_remote_cache_enabled = self._get_env_bool('TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE', False)
        self.autotune_remote_cache_enabled = self._get_env_bool('TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE', False)
        self.cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR', None)
        self.force_disable_caches = self._get_env_bool('TORCHINDUCTOR_FORCE_DISABLE_CACHES', False)
        
        # Check if torch.compile is supported on the current device
        if not self.is_compile_supported():
            logger.warning("torch.compile is not supported on this device. Compilation will be disabled.")
            self.compile_enabled = False
        
        self._configure_caches()
        
        self.logging_enabled = self.is_compile_supported()
        self.logging_options = {}

    def _get_env_bool(self, var_name, default=False):
        return os.environ.get(var_name, str(default)).lower() in ('1', 'true', 'yes', 'on')

    def _configure_caches(self):
        if self.force_disable_caches:
            logger.info("Forcibly disabling all caches as per TORCHINDUCTOR_FORCE_DISABLE_CACHES")
            self.fx_graph_cache_enabled = False
            self.fx_graph_remote_cache_enabled = False
            self.autotune_remote_cache_enabled = False
        else:
            if self.cache_dir:
                os.environ['TORCHINDUCTOR_CACHE_DIR'] = self.cache_dir
                logger.info(f"Set TORCHINDUCTOR_CACHE_DIR to {self.cache_dir}")
            
            if self.fx_graph_remote_cache_enabled or self.autotune_remote_cache_enabled:
                redis_host = os.environ.get('TORCHINDUCTOR_REDIS_HOST', 'localhost')
                redis_port = os.environ.get('TORCHINDUCTOR_REDIS_PORT', '6379')
                logger.info(f"Configured Redis cache at {redis_host}:{redis_port}")

    def is_compile_supported(self):
        """Check if the current CUDA device supports torch.compile"""
        return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0)

    def configure_logging(self, **kwargs):
        """
        Configure logging options for torch.compile.
        
        :param kwargs: Logging options to enable. Valid keys are:
                       dynamo, graph, fusion, output_code
        """
        if not self.logging_enabled:
            logger.warning("Logging is not enabled as torch.compile is not supported on this device.")
            return

        valid_options = {'dynamo', 'graph', 'fusion', 'output_code'}
        self.logging_options = {k: v for k, v in kwargs.items() if k in valid_options}
        
        torch._logging.set_logs(**self.logging_options)
        logger.info(f"Configured logging options: {self.logging_options}")

    def optimize(self, obj):
        if not self.compile_enabled:
            return obj
        
        try:
            # Configure caching options
            torch._inductor.config.fx_graph_cache = self.fx_graph_cache_enabled
            torch._inductor.config.fx_graph_remote_cache = self.fx_graph_remote_cache_enabled
            torch._inductor.config.autotune_remote_cache = self.autotune_remote_cache_enabled
            
            # Apply logging configuration
            if self.logging_enabled:
                torch._logging.set_logs(**self.logging_options)
            
            return torch.compile(obj, mode=self.compile_mode, fullgraph=self.fullgraph)
        except Exception as e:
            logger.error(f"Compilation failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return obj
    
    def optimize_optimizer_with_scheduler(self, optimizer, scheduler):
        """
        Optimize the step() methods of both the optimizer and scheduler using torch.compile.
        """
        if not self.compile_enabled:
            return optimizer, scheduler
        try:
            # Ensure the learning rate is wrapped in a tensor
            for param_group in optimizer.param_groups:
                if not isinstance(param_group['lr'], torch.Tensor):
                    param_group['lr'] = torch.tensor(param_group['lr'], device=optimizer.param_groups[0]['params'][0].device)
            
            # Configure caching options
            torch._inductor.config.fx_graph_cache = self.fx_graph_cache_enabled
            torch._inductor.config.fx_graph_remote_cache = self.fx_graph_remote_cache_enabled
            torch._inductor.config.autotune_remote_cache = self.autotune_remote_cache_enabled
            
            @torch.compile(mode=self.compile_mode, fullgraph=False)
            def optimized_step():
                optimizer.step()
                scheduler.step()
            
            # Replace the original step methods with the optimized one
            optimizer.step = optimized_step
            scheduler.step = lambda: None  # Scheduler step is now included in the optimized_step
            return optimizer, scheduler
        except Exception as e:
            logger.error(f"Optimizer and scheduler compilation failed: {str(e)}")
            logger.debug(traceback.format_exc())
            return optimizer, scheduler

    def benchmark_torch_function_in_microseconds(self, f, *args, **kwargs):
        t0 = benchmark.Timer(
            stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
        )
        return t0.blocked_autorange().mean * 1e6

    def benchmark(self, original_fn, optimized_fn, inputs, n_iters=10):
        logger.info("Benchmarking original vs optimized")
        
        if self.logging_enabled:
            logger.info("Logging is enabled. Performance may be affected.")
            torch._logging.set_logs(**self.logging_options)
        
        original_times = []
        optimized_times = []
        
        for _ in tqdm(range(n_iters)):
            orig_time = self.benchmark_torch_function_in_microseconds(original_fn, *inputs)
            original_times.append(orig_time)
            
            opt_time = self.benchmark_torch_function_in_microseconds(optimized_fn, *inputs)
            optimized_times.append(opt_time)
        
        avg_original = sum(original_times) / len(original_times)
        avg_optimized = sum(optimized_times) / len(optimized_times)
        
        speedup = avg_original / avg_optimized
        
        logger.info(f"Average original time: {avg_original:.4f}us")
        logger.info(f"Average optimized time: {avg_optimized:.4f}us")
        logger.info(f"Speedup: {speedup:.2f}x")

    def analyze(self, fn, *args):
        torch._dynamo.reset()
        explain_output = torch._dynamo.explain(fn)(*args)
        logger.info("TorchDynamo Explanation:")
        logger.info(explain_output)

    def set_cache_dir(self, directory):
        """Set the cache directory for Inductor"""
        self.cache_dir = directory
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = directory
        logger.info(f"Set TORCHINDUCTOR_CACHE_DIR to {directory}")

    def force_disable_all_caches(self):
        """Force disable all caches for debugging purposes"""
        self.force_disable_caches = True
        self.fx_graph_cache_enabled = False
        self.fx_graph_remote_cache_enabled = False
        self.autotune_remote_cache_enabled = False
        os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
        logger.info("Forcibly disabled all caches")

    def display_compilation_info(self, fn, *args):
        """
        Display detailed information about the compilation process.
        
        :param fn: The function to analyze
        :param args: Arguments to pass to the function
        """
        if not self.logging_enabled:
            logger.warning("Compilation info display is not available as torch.compile is not supported on this device.")
            return

        torch._dynamo.reset()
        
        logger.info("Displaying Dynamo tracing information:")
        torch._logging.set_logs(dynamo=logging.DEBUG)
        fn(*args)
        
        logger.info("Displaying traced graph:")
        torch._logging.set_logs(graph=True)
        fn(*args)
        
        logger.info("Displaying fusion decisions:")
        torch._logging.set_logs(fusion=True)
        fn(*args)
        
        logger.info("Displaying output code:")
        torch._logging.set_logs(output_code=True)
        fn(*args)
        
        # Reset logging options
        torch._logging.set_logs(**self.logging_options)