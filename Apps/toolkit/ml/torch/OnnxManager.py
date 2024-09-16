# My code starts from here
import torch
import onnx
import onnxruntime
import numpy as np
from typing import List, Tuple, Union, Dict, Any
from onnx import shape_inference, optimizer, checker, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from onnx.helper import (
    make_model, make_node, make_graph, make_tensor_value_info,
    make_function, make_opsetid
)

class OnnxManager:
    def __init__(self) -> None:
        self.model = None
        self.session = None
        self.ref_evaluator = None
        self.use_onnx_runtime = False  # Feature flag for ONNX Runtime
        self.use_onnx_runtime = False  # Feature flag for ONNX Runtime
        self.target_opset = None  # New: Target opset version
        self.enable_converters = False  # Feature flag for converter functionality

    def load_model(self, model_path: str, target_opset: int = None) -> None:
        """Load an ONNX model from a file."""
        self.model = onnx.load(model_path)
        self.target_opset = target_opset or self._get_model_opset()
        if self.use_onnx_runtime:
            self.session = onnxruntime.InferenceSession(model_path)
        else:
            self.ref_evaluator = ReferenceEvaluator(self.model)

    def create_onnx_function(self, domain: str, name: str, inputs: List[str], 
                             outputs: List[str], nodes: List[onnx.NodeProto], 
                             opset_imports: List[onnx.OperatorSetIdProto]) -> onnx.FunctionProto:
        """Create an ONNX function."""
        return make_function(domain, name, inputs, outputs, nodes, opset_imports)
    
    def infer(self, input_data: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Perform inference using the loaded ONNX model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        if self.use_onnx_runtime:
            return self._infer_onnxruntime(input_data)
        else:
            return self._infer_pytorch(input_data)

    def _infer_onnxruntime(self, input_data: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Perform inference using ONNX Runtime."""
        if isinstance(input_data, torch.Tensor):
            input_data = [input_data]
        
        input_feed = {input.name: data.numpy() for input, data in zip(self.session.get_inputs(), input_data)}
        outputs = self.session.run(None, input_feed)
        return [torch.from_numpy(output) for output in outputs]

    def _infer_pytorch(self, input_data: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Perform inference using PyTorch."""
        if isinstance(input_data, torch.Tensor):
            input_data = [input_data]
        
        input_feed = {input.name: data.numpy() for input, data in zip(self.model.graph.input, input_data)}
        
        outputs = self.ref_evaluator.run(None, input_feed)
        
        return [torch.from_numpy(output) for output in outputs]

    @staticmethod
    def pytorch_to_onnx(model: torch.nn.Module, input_shape: Tuple[int, ...], output_path: str, opset_version: int = None) -> None:
        """Convert a PyTorch model to ONNX format."""
        dummy_input = torch.randn(input_shape)
        torch.onnx.export(model, dummy_input, output_path, opset_version=opset_version)

    def get_model_info(self) -> dict:
        """Return information about the loaded ONNX model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        return {
            "ir_version": self.model.ir_version,
            "producer_name": self.model.producer_name,
            "producer_version": self.model.producer_version,
            "domain": self.model.domain,
            "model_version": self.model.model_version,
            "doc_string": self.model.doc_string,
            "opset_version": self.target_opset,
        }

    def _get_model_opset(self) -> int:
        """Get the opset version of the loaded model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        return max(opset.version for opset in self.model.opset_import)
    
    def update_opset_version(self, new_opset: int):
        """Update the opset version of the model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        self.target_opset = new_opset
        for opset in self.model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                opset.version = new_opset
        
        # Re-initialize the reference evaluator with the updated model
        self.ref_evaluator = ReferenceEvaluator(self.model)
    
    def get_intermediate_outputs(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get intermediate outputs for all nodes in the model."""
        if self.model is None or self.ref_evaluator is None:
            raise ValueError("No model loaded or reference evaluator not initialized.")
        
        intermediate_outputs = {}
        for node in self.model.graph.node:
            outputs = self.ref_evaluator.run(node.output, input_data)
            for output_name, output_value in zip(node.output, outputs):
                intermediate_outputs[output_name] = output_value
        
        return intermediate_outputs
    
    def visualize_model(self) -> None:
        """Visualize the ONNX model using netron."""
        print("To visualize the model, please use netron: https://github.com/lutzroeder/netron")

    def enable_onnx_runtime(self, enable: bool = True) -> None:
        """Enable or disable the use of ONNX Runtime for inference."""
        self.use_onnx_runtime = enable

    def infer_shapes(self):
        """Infer and update shapes of the model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        self.model = shape_inference.infer_shapes(self.model)

    def optimize_model(self, passes: List[str] = None):
        """Perform advanced model optimization."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if passes is None:
            passes = ['eliminate_identity', 'eliminate_nop_transpose', 'fuse_consecutive_squeezes',
            'fuse_consecutive_reduce_unsqueeze', 'fuse_bn_into_conv']
        
        self.model = optimizer.optimize(self.model, passes)

    def load_model_with_custom_ops(self, model_path: str, custom_ops: List[OpRun]):
        """Load an ONNX model with custom operators."""
        self.model = onnx.load(model_path)
        if self.use_onnx_runtime:
            self.session = onnxruntime.InferenceSession(model_path)
        else:
            self.ref_evaluator = ReferenceEvaluator(self.model, new_ops=custom_ops)

    def check_model(self):
        """Check the validity of the loaded ONNX model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        checker.check_model(self.model)

    def check_and_infer_shapes(self):
        """Check the model and infer shapes."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            checker.check_model(self.model)
            self.model = shape_inference.infer_shapes(self.model)
        except Exception as e:
            raise ValueError(f"Model check or shape inference failed: {str(e)}")

    def serialize_tensor(self, tensor: np.ndarray, name: str) -> bytes:
        """Serialize a numpy tensor to ONNX format."""
        onnx_tensor = numpy_helper.from_array(tensor, name=name)
        return onnx_tensor.SerializeToString()
    
    def deserialize_tensor(self, serialized_tensor: bytes) -> np.ndarray:
        """Deserialize an ONNX tensor to numpy array."""
        onnx_tensor = onnx.TensorProto()
        onnx_tensor.ParseFromString(serialized_tensor)
        return numpy_helper.to_array(onnx_tensor)
    
    def serialize_model(self) -> bytes:
        """Serialize the loaded ONNX model to bytes."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        return self.model.SerializeToString()

    def deserialize_model(self, serialized_model: bytes):
        """Deserialize and load an ONNX model from bytes."""
        self.model = onnx.load_from_string(serialized_model)
        if self.use_onnx_runtime:
            self.session = onnxruntime.InferenceSession(serialized_model)
        else:
            self.ref_evaluator = ReferenceEvaluator(self.model)

    def create_if_node(self, condition: str, then_branch: onnx.GraphProto, 
                       else_branch: onnx.GraphProto, outputs: List[str]) -> onnx.NodeProto:
        """Create an If node with then and else branches."""
        return make_node("If", [condition], outputs, 
                         then_branch=then_branch, else_branch=else_branch)

    def create_scan_node(self, body: onnx.GraphProto, inputs: List[str], 
                         outputs: List[str], num_scan_inputs: int) -> onnx.NodeProto:
        """Create a Scan node with a subgraph body."""
        return make_node("Scan", inputs, outputs, body=body, num_scan_inputs=num_scan_inputs)

    def create_onnx_graph(self, nodes: List[onnx.NodeProto], name: str, 
                          inputs: List[onnx.ValueInfoProto], outputs: List[onnx.ValueInfoProto], 
                          initializers: List[onnx.TensorProto] = None) -> onnx.GraphProto:
        """Create an ONNX graph programmatically."""
        return make_graph(nodes, name, inputs, outputs, initializers)

    def add_initializer(self, name: str, tensor: np.ndarray):
        """Add an initializer to the model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        initializer = numpy_helper.from_array(tensor, name=name)
        self.model.graph.initializer.append(initializer)

    def add_custom_op(self, op: OpRun):
        """Add a custom operator to the model."""
        if self.ref_evaluator is None:
            raise ValueError("Reference evaluator not initialized. Load a model first.")
        
        self.ref_evaluator.add_op(op)
    
    def enable_converters(self, enable: bool = True):
        """Enable or disable converter functionality."""
        self.enable_converters = enable

    def benchmark_inference(self, input_data: Union[torch.Tensor, List[torch.Tensor]], num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        import time

        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        total_time = 0
        for _ in range(num_runs):
            start_time = time.time()
            self.infer(input_data)
            total_time += time.time() - start_time

        avg_time = total_time / num_runs
        return {
            "average_inference_time": avg_time,
            "inference_per_second": 1 / avg_time
        }

