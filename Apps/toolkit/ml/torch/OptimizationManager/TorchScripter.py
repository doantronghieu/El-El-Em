# My code starts from here
import torch
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import time
import torch.nn.functional as F

class TorchScripter:
    def __init__(self):
        self.compiled_modules = {}
        self.use_mixed_compilation = False
        self.compilation_method = "auto"  # Can be "auto", "trace", or "script"
        self.enable_cpp_code_generation = False 
        self.enable_sanity_check = False
        self.enable_detailed_performance_analysis = False
        self.enable_freezing = False
        self.disable_jit = False
        self.enable_dynamic_parallelism = False
        
    def compile_module(self, module: torch.nn.Module, example_inputs: Any) -> torch.jit.ScriptModule:
        """
        Compile a PyTorch module using TorchScript.
        
        Args:
            module (torch.nn.Module): The PyTorch module to compile.
            example_inputs (Any): Example inputs to use for tracing.
        
        Returns:
            torch.jit.ScriptModule: The compiled TorchScript module.
        """
        if self.disable_jit:
            print("JIT compilation disabled. Returning original module.")
            return module
          
        try:
            if self.compilation_method == "auto":
                compiled_module = self._compile_auto(module, example_inputs)
            elif self.compilation_method == "trace":
                compiled_module = torch.jit.trace(module, example_inputs)
            elif self.compilation_method == "script":
                compiled_module = torch.jit.script(module)
            elif self.compilation_method == "trace_module":
                compiled_module = self.trace_module(module, {"forward": example_inputs})
            else:
                raise ValueError(f"Invalid compilation method: {self.compilation_method}")
              
            if self.enable_dynamic_parallelism:
                compiled_module = self.add_fork_wait_support(compiled_module)
                
            if self.use_mixed_compilation:
                compiled_module = self.apply_mixed_compilation(compiled_module)
            
            self.compiled_modules[type(module).__name__] = compiled_module
            
            if self.enable_sanity_check:
                self.sanity_check(module, compiled_module, example_inputs)
            
            return compiled_module
        except Exception as e:
            print(f"Compilation failed: {str(e)}")
            self.print_graph(module)  # Print graph for debugging
            raise

    def _compile_auto(self, module: torch.nn.Module, example_inputs: Any) -> torch.jit.ScriptModule:
        """Helper method for auto compilation"""
        try:
            return torch.jit.script(module)
        except Exception as e:
            print(f"Scripting failed, falling back to tracing. Error: {str(e)}")
            return torch.jit.trace(module, example_inputs)

    def sanity_check(self, original_module: torch.nn.Module, compiled_module: torch.jit.ScriptModule, example_inputs: Any):
        """
        Perform a sanity check to ensure the compiled module produces the same output as the original module.
        
        Args:
            original_module (torch.nn.Module): The original PyTorch module.
            compiled_module (torch.jit.ScriptModule): The compiled TorchScript module.
            example_inputs (Any): Example inputs to use for the sanity check.
        """
        with torch.no_grad():
            original_output = original_module(example_inputs)
            compiled_output = compiled_module(example_inputs)

        original_top5 = F.softmax(original_output, dim=1).topk(5)
        compiled_top5 = F.softmax(compiled_output, dim=1).topk(5)

        if torch.allclose(original_top5.values, compiled_top5.values, atol=1e-3) and torch.equal(original_top5.indices, compiled_top5.indices):
            print("Sanity check passed: Original and compiled models produce similar top 5 results.")
        else:
            print("Warning: Sanity check failed. Original and compiled models produce different top 5 results.")
            print(f"Original model top 5 results:\n  {original_top5}")
            print(f"Compiled model top 5 results:\n  {compiled_top5}")

    def add_fork_wait_support(self, module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Add support for dynamic parallelism using fork and wait.
        
        Args:
            module (torch.jit.ScriptModule): The module to add fork and wait support to.
        
        Returns:
            torch.jit.ScriptModule: The module with fork and wait support.
        """
        def fork_wrapper(func):
            return torch.jit.fork(func)

        def wait_wrapper(future):
            return torch.jit.wait(future)

        module.define("fork_wrapper", fork_wrapper)
        module.define("wait_wrapper", wait_wrapper)
        
        return module
    
    def freeze_module(self, module: torch.jit.ScriptModule, preserve_methods: List[str] = []) -> torch.jit.ScriptModule:
        """
        Freeze a ScriptModule, inlining all submodules, parameters, and attributes.
        
        Args:
            module (torch.jit.ScriptModule): The module to freeze.
            preserve_methods (List[str]): List of method names to preserve during freezing.
        
        Returns:
            torch.jit.ScriptModule: The frozen module.
        """
        if not self.enable_freezing:
            print("Warning: Freezing is disabled. Enable it by setting enable_freezing to True.")
            return module
        return torch.jit.freeze(module, reserve_methods=preserve_methods)
    
    def optimize_module(self, module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Apply various optimization techniques to the module, including freezing if enabled.
        
        Args:
            module (torch.jit.ScriptModule): The module to optimize.
        
        Returns:
            torch.jit.ScriptModule: The optimized module.
        """
        # Optimize for inference
        module = torch.jit.optimize_for_inference(module)
        
        # Freeze the module if enabled
        if self.enable_freezing:
            module = self.freeze_module(module)
        
        # Enable oneDNN fusion
        torch.jit.enable_onednn_fusion(True)
        
        return module

    def save_compiled_module(self, module_name: str, path: str) -> None:
        """
        Save a compiled module to disk.
        
        Args:
            module_name (str): Name of the compiled module to save.
            path (str): Path to save the module to.
        """
        if module_name not in self.compiled_modules:
            raise ValueError(f"No compiled module named {module_name}")
        
        torch.jit.save(self.compiled_modules[module_name], path)

    def load_compiled_module(self, path: str) -> torch.jit.ScriptModule:
        """
        Load a compiled module from disk.
        
        Args:
            path (str): Path to load the module from.
        
        Returns:
            torch.jit.ScriptModule: The loaded TorchScript module.
        """
        return torch.jit.load(path)

    def set_compilation_method(self, method: str):
        if method not in ["auto", "trace", "script", "trace_module"]:
            raise ValueError(f"Invalid compilation method: {method}")
        self.compilation_method = method

    @staticmethod
    def optimize_for_inference(module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Optimize a compiled module for inference.
        
        Args:
            module (torch.jit.ScriptModule): The module to optimize.
        
        Returns:
            torch.jit.ScriptModule: The optimized module.
        """
        return torch.jit.optimize_for_inference(module)

    @staticmethod
    def set_fusion_strategy(strategy: List[Tuple[str, int]]) -> None:
        """
        Set the fusion strategy for TorchScript compilation.
        
        Args:
            strategy (List[Tuple[str, int]]): List of (fusion_name, max_fused_kernel_size) pairs.
        """
        torch.jit.set_fusion_strategy(strategy)

    @staticmethod
    def enable_onednn_fusion(enabled: bool = True) -> None:
        """
        Enable or disable oneDNN JIT fusion.
        
        Args:
            enabled (bool): Whether to enable oneDNN fusion.
        """
        torch.jit.enable_onednn_fusion(enabled)

    def enable_mixed_compilation(self, enabled: bool = True):
        """
        Enable or disable mixed compilation.
        
        Args:
            enabled (bool): Whether to enable mixed compilation.
        """
        self.use_mixed_compilation = enabled

    @staticmethod
    def freeze(module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Freeze a ScriptModule, inlining all submodules, parameters, and attributes.
        
        Args:
            module (torch.jit.ScriptModule): The module to freeze.
        
        Returns:
            torch.jit.ScriptModule: The frozen module.
        """
        return torch.jit.freeze(module)

    @staticmethod
    def trace_module(mod: torch.nn.Module, 
                     inputs: Dict[str, Any], 
                     check_trace: bool = True,
                     check_inputs: Optional[List[Dict[str, Any]]] = None) -> torch.jit.ScriptModule:
        """
        Trace a module with multiple methods.
        
        Args:
            mod (torch.nn.Module): The module to trace.
            inputs (Dict[str, Any]): A dict of sample inputs.
            check_trace (bool): Whether to check the trace for correctness.
            check_inputs (Optional[List[Dict[str, Any]]]): Additional sample inputs for checking.
        
        Returns:
            torch.jit.ScriptModule: The traced module.
        """
        return torch.jit.trace_module(mod, inputs, check_trace, check_inputs)
    
    def apply_mixed_compilation(self, module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Apply mixed compilation to a module, combining tracing and scripting.
        
        Args:
            module (torch.jit.ScriptModule): The module to apply mixed compilation to.
        
        Returns:
            torch.jit.ScriptModule: The module with mixed compilation applied.
        """
        # This is a placeholder implementation. In a real-world scenario,
        # you would need to implement logic to determine which parts of the
        # module should be traced and which should be scripted.
        return module

    def parallelize_module(self, module: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
        """
        Attempt to automatically parallelize suitable parts of a module.
        
        Args:
            module (torch.jit.ScriptModule): The module to parallelize.
        
        Returns:
            torch.jit.ScriptModule: The parallelized module.
        """
        # This is a placeholder implementation. In a real-world scenario,
        # you would need to implement logic to identify parallelizable parts
        # of the module and apply fork and wait appropriately.
        return module
    
    def analyze_performance(self, module: torch.jit.ScriptModule, inputs: Any) -> Dict[str, Dict[str, float]]:
        """
        Analyze the performance of a compiled module, including parallel execution if enabled.
        
        Args:
            module (torch.jit.ScriptModule): The compiled module to analyze.
            inputs (Any): Inputs to use for performance testing.
        
        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing performance metrics for different versions of the module.
        """
        metrics = {}
        
        # Analyze non-parallelized module
        metrics["original"] = self._run_performance_analysis(module, inputs)
        
        # Analyze frozen module if freezing is enabled
        if self.enable_freezing:
            frozen_module = self.freeze_module(module)
            metrics["frozen"] = self._run_performance_analysis(frozen_module, inputs)
        
        # Analyze parallelized module if dynamic parallelism is enabled
        if self.enable_dynamic_parallelism:
            parallelized_module = self.parallelize_module(module)
            metrics["parallelized"] = self._run_performance_analysis(parallelized_module, inputs)
        
        return metrics

    def enable_dynamic_parallelism(self, enabled: bool = True):
        """
        Enable or disable dynamic parallelism support.
        
        Args:
            enabled (bool): Whether to enable dynamic parallelism.
        """
        self.enable_dynamic_parallelism = enabled

    def _run_performance_analysis(self, module: torch.jit.ScriptModule, inputs: Any) -> Dict[str, float]:
        """Helper method to run performance analysis on a module"""
        module_metrics = {}
        
        module(*inputs)  # Warm-up run
        
        start_time = time.time()
        num_runs = 100
        for _ in range(num_runs):
            output = module(*inputs)
        end_time = time.time()
        
        module_metrics["average_execution_time"] = (end_time - start_time) / num_runs
        module_metrics["output_size"] = sum(o.numel() for o in output) if isinstance(output, tuple) else output.numel()
        
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            output = module(*inputs)
            module_metrics["peak_gpu_memory_usage"] = torch.cuda.max_memory_allocated() / 1024 / 1024  # in MB
        
        if self.enable_detailed_performance_analysis:
            module_metrics["flops"] = self._estimate_flops(module, inputs)
            module_metrics["parameter_count"] = sum(p.numel() for p in module.parameters())
            module_metrics["trainable_parameter_count"] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return module_metrics

    def generate_cpp_loading_code(self, module_name: str, path: str) -> str:
        if not self.enable_cpp_code_generation:
            raise ValueError("C++ code generation is not enabled. Set enable_cpp_code_generation to True.")
        
        return f"""
        #include <torch/script.h>
        #include <torch/nn/functional/activation.h>
        #include <iostream>

        int main() {{
            torch::jit::script::Module module;
            try {{
                module = torch::jit::load("{path}");
                std::cout << "Model {module_name} loaded successfully\\n";
            }}
            catch (const c10::Error& e) {{
                std::cerr << "Error loading the model\\n";
                std::cerr << e.what() << std::endl;
                return -1;
            }}
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(torch::rand({{1, 3, 224, 224}}));

            torch::NoGradGuard no_grad;
            module.eval();
            at::Tensor output = module.forward(inputs).toTensor();

            namespace F = torch::nn::functional;
            at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
            std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
            at::Tensor top5 = std::get<1>(top5_tensor);

            std::cout << "Top 5 predictions:\\n" << top5[0] << std::endl;

            return 0;
        }}
        """

    def _estimate_flops(self, module: torch.jit.ScriptModule, inputs: Any) -> int:
        def count_convolutions(mod):
            flops = 0
            for m in mod.modules():
                if isinstance(m, torch.nn.Conv2d):
                    flops += m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * \
                             inputs[0].size()[2] * inputs[0].size()[3] / m.stride[0] / m.stride[1]
            return flops

        return count_convolutions(module)

    def print_graph(self, module: torch.jit.ScriptModule):
        """Print the graph of a TorchScript module for debugging."""
        print(module.graph)

    def disable_jit_compilation(self, disable: bool = True):
        """Disable JIT compilation for debugging purposes."""
        self.disable_jit = disable
        if disable:
            print("JIT compilation disabled. Use for debugging only.")
        else:
            print("JIT compilation enabled.")

    def export_settings(self) -> Dict[str, Any]:
        """Export current compilation and optimization settings."""
        return {
            "compilation_method": self.compilation_method,
            "use_mixed_compilation": self.use_mixed_compilation,
            "enable_freezing": self.enable_freezing,
            "enable_cpp_code_generation": self.enable_cpp_code_generation,
            "enable_sanity_check": self.enable_sanity_check,
            "enable_detailed_performance_analysis": self.enable_detailed_performance_analysis,
            "disable_jit": self.disable_jit
        }

    @staticmethod
    def create_torchscript_class(class_def: type) -> torch.jit.ScriptModule:
        """Create a TorchScript class from a Python class definition."""
        return torch.jit.script(class_def)

    @staticmethod
    def add_builtin_function(module: torch.jit.ScriptModule, func_name: str, func: Callable):
        """Add a built-in function to a TorchScript module."""
        torch.jit.script_if_tracing(func)
        setattr(module, func_name, func)