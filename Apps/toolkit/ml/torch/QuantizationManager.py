import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Type
import yaml
from tqdm import tqdm
from loguru import logger
import copy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

from torch.ao.quantization import (
    QConfigMapping, default_dynamic_qconfig, get_default_qconfig,
    quantize_dynamic, get_default_qat_qconfig, DeQuantStub, QuantStub
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

config = Config()

torch.backends.quantized.engine = config.get('quantization_backend', 'qnnpack')

logger.add(config.get('log_file', 'quantization.log'), rotation=config.get('log_rotation', '500 MB'))

class AbstractModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def modify_for_quantization(self) -> None:
        pass

class AbstractDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pass

class QuantizationWrapper:
    def __init__(self):
        self.qconfig_mapping = QConfigMapping()
        self.quantizer = None
        
    def set_global_qconfig(self, backend: str = 'qnnpack') -> None:
        try:
            qconfig = get_default_qconfig(backend)
            self.qconfig_mapping.set_global(qconfig)
        except ValueError as e:
            logger.error(f"Error setting global qconfig: {str(e)}")
            raise
        
    def set_qconfig_for_module(self, module_type: Type[nn.Module], qconfig: Any) -> None:
        self.qconfig_mapping.set_object_type(module_type, qconfig)
        
    def set_quantizer(self, backend: str = 'xnnpack') -> None:
        if backend == 'xnnpack':
            self.quantizer = XNNPACKQuantizer()
            self.quantizer.set_global(get_symmetric_quantization_config())
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
    def prepare_fx(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model.eval()
        prepared_model = prepare_fx(model, self.qconfig_mapping, example_inputs)
        return prepared_model
    
    def prepare_qat_fx(self, model: nn.Module, example_inputs: Any) -> nn.Module:
        model.train()
        prepared_model = prepare_qat_fx(model, self.qconfig_mapping, example_inputs)
        return prepared_model
    
    def convert_fx(self, prepared_model: nn.Module) -> nn.Module:
        quantized_model = convert_fx(prepared_model)
        return quantized_model
    
    def quantize_dynamic(self, model: nn.Module, qconfig_spec: Optional[Dict[Type[nn.Module], Any]] = None) -> nn.Module:
        if qconfig_spec is None:
            qconfig_spec = {nn.Linear, nn.LSTM}
        return quantize_dynamic(model, qconfig_spec, dtype=torch.qint8)

class QuantizationCalibrator:
    def __init__(self, model: nn.Module, data_loader: DataLoader):
        self.model = model
        self.data_loader = data_loader
        
    def calibrate(self, num_batches: Optional[int] = None) -> None:
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(tqdm(self.data_loader, desc="Calibrating")):
                if num_batches is not None and i >= num_batches:
                    break
                self.model(inputs)
        logger.info(f"Calibration completed using {i+1} batches.")

class QuantizationDebugger:
    def __init__(self, float_model: nn.Module, quantized_model: nn.Module):
        self.float_model = float_model
        self.quantized_model = quantized_model
        
    def compare_outputs(self, inputs: torch.Tensor) -> None:
        with torch.no_grad():
            float_output = self.float_model(inputs)
            quantized_output = self.quantized_model(inputs)
        
        diff = torch.abs(float_output - quantized_output)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        logger.info(f"Max absolute difference: {max_diff.item()}")
        logger.info(f"Mean absolute difference: {mean_diff.item()}")
        
    def print_model_size(self) -> None:
        def get_size(model: nn.Module) -> float:
            torch.save(model.state_dict(), "temp.p")
            size = os.path.getsize("temp.p") / (1024 * 1024)
            os.remove("temp.p")
            return size

        float_size = get_size(self.float_model)
        quantized_size = get_size(self.quantized_model)

        logger.info(f"Float model size: {float_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Size reduction: {(1 - quantized_size/float_size)*100:.2f}%")

class QuantizationManager:
    def __init__(self, model: AbstractModel, example_inputs: torch.Tensor, data_loader: DataLoader):
        self.original_model = model
        self.example_inputs = example_inputs
        self.data_loader = data_loader
        self.wrapper = QuantizationWrapper()
        
    def run_static_quantization(self) -> nn.Module:
        logger.info("Starting static quantization...")
        self.wrapper.set_global_qconfig(config.get('quantization_backend', 'qnnpack'))
        self.wrapper.set_quantizer()
        
        model_copy = copy.deepcopy(self.original_model)
        model_copy.modify_for_quantization()
        prepared_model = self.wrapper.prepare_fx(model_copy, self.example_inputs)
        
        calibrator = QuantizationCalibrator(prepared_model, self.data_loader)
        calibrator.calibrate(num_batches=config.get('static_quantization_calibration_batches', 10))
        
        quantized_model = self.wrapper.convert_fx(prepared_model)
        
        logger.info("Static quantization completed.")
        return quantized_model
    
    def run_qat(self, num_epochs: int = 5) -> nn.Module:
        logger.info("Starting quantization-aware training...")
        self.wrapper.set_global_qconfig(config.get('quantization_backend', 'qnnpack'))
        
        model_to_quantize = copy.deepcopy(self.original_model)
        model_to_quantize.modify_for_quantization()
        model_to_quantize.train()
        
        qconfig = get_default_qat_qconfig(config.get('quantization_backend', 'qnnpack'))
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        prepared_model = self.wrapper.prepare_qat_fx(model_to_quantize, self.example_inputs)
        
        optimizer = torch.optim.Adam(prepared_model.parameters(), lr=config.get('qat_learning_rate', 0.0001))
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for inputs, targets in tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                outputs = prepared_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            logger.info(f"Completed epoch {epoch+1}/{num_epochs}")
        
        quantized_model = self.wrapper.convert_fx(prepared_model.eval())
        logger.info("Quantization-aware training completed.")
        return quantized_model
    
    def run_dynamic_quantization(self) -> nn.Module:
        logger.info("Starting dynamic quantization...")
        model_copy = copy.deepcopy(self.original_model)
        model_copy.modify_for_quantization()
        quantized_model = self.wrapper.quantize_dynamic(model_copy)
        logger.info("Dynamic quantization completed.")
        return quantized_model
    
    def save_quantized_model(self, model: nn.Module, path: str) -> None:
        torch.jit.save(torch.jit.script(model), path)
        logger.info(f"Quantized model saved to {path}")
      
    def load_quantized_model(self, path: str) -> nn.Module:
        model = torch.jit.load(path)
        logger.info(f"Quantized model loaded from {path}")
        return model
  
    def benchmark_performance(self, model: nn.Module, num_runs: int = 100) -> Tuple[float, float]:
        model.eval()
        times = []
        
        # Warm-up run
        _ = model(self.example_inputs)
        
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start_time = time.time()
            with torch.no_grad():
                _ = model(self.example_inputs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        logger.info(f"Average inference time: {avg_time*1000:.2f} ms")
        logger.info(f"Standard deviation: {std_dev*1000:.2f} ms")
        
        return avg_time, std_dev

    def compare_models(self, float_model: nn.Module, quantized_model: nn.Module) -> None:
        debugger = QuantizationDebugger(float_model, quantized_model)
        debugger.print_model_size()
        
        logger.info("Benchmarking float model:")
        float_time, _ = self.benchmark_performance(float_model)
        
        logger.info("Benchmarking quantized model:")
        quant_time, _ = self.benchmark_performance(quantized_model)
        
        speedup = float_time / quant_time
        logger.info(f"Speedup: {speedup:.2f}x")

    def validate_accuracy(self, model: nn.Module, data_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="Validating"):
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        logger.info(f"Validation Accuracy: {accuracy*100:.2f}%")
        return accuracy

class YourModel(AbstractModel):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.is_modified = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def modify_for_quantization(self) -> None:
        if not self.is_modified:
            self.model.conv1 = nn.Sequential(
                self.model.conv1,
                nn.BatchNorm2d(self.model.conv1.out_channels),
                nn.ReLU()
            )
            for name, module in self.model.named_children():
                if isinstance(module, models.resnet.BasicBlock):
                    module.conv1 = nn.Sequential(
                        module.conv1,
                        nn.BatchNorm2d(module.conv1.out_channels),
                        nn.ReLU()
                    )
                    module.conv2 = nn.Sequential(
                        module.conv2,
                        nn.BatchNorm2d(module.conv2.out_channels)
                    )
            self.is_modified = True
        else:
            logger.info("Model has already been modified for quantization.")
            
class YourDataset(AbstractDataset):
    def __init__(self, size: int = 1000, img_size: int = 224):
        super().__init__()
        self.size = size
        self.img_size = img_size
        self.data = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.colors = [
            (0, 0, 255),    # Blue
            (0, 255, 0),    # Green
            (255, 0, 0),    # Red
            (255, 255, 0),  # Yellow
            (0, 0, 0),      # Black
            (255, 255, 255),# White
            (128, 128, 128),# Gray
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (165, 42, 42)   # Brown
        ]
        
        self._generate_data()
    
    def _generate_data(self):
        for _ in range(self.size):
            label = np.random.randint(0, 10)  # Random label 0 to 9
            color = self.colors[label]
            
            # Create a solid color image
            img = Image.new('RGB', (self.img_size, self.img_size), color)
            
            # Add some noise to make it more realistic
            img_array = np.array(img)
            noise = np.random.randint(-20, 20, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            self.data.append(img)
            self.labels.append(label)
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def create_model(model_type: str, num_classes: int, pretrained: bool) -> AbstractModel:
    if model_type.lower() == 'resnet18':
        return YourModel(num_classes, pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def main():
    # Create datasets and data loaders
    train_dataset = YourDataset(size=config.get('train_dataset_size', 1000), img_size=config.get('img_size', 224))
    val_dataset = YourDataset(size=config.get('val_dataset_size', 200), img_size=config.get('img_size', 224))
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False)

    # Create your model
    model_type = config.get('model_type', 'resnet18')
    num_classes = config.get('num_classes', 10)
    pretrained = config.get('weights', 'IMAGENET1K_V1') is not None
    model = create_model(model_type, num_classes, pretrained)

    # Prepare example inputs
    example_inputs = torch.randn(1, 3, config.get('img_size', 224), config.get('img_size', 224))

    # Initialize the QuantizationManager
    qmanager = QuantizationManager(model, example_inputs, train_loader)

    # Run static quantization
    static_quantized_model = qmanager.run_static_quantization()

    # Save the static quantized model
    qmanager.save_quantized_model(static_quantized_model, config.get('static_quantized_model_path', 'static_quantized_model.pt'))

    # Run dynamic quantization
    dynamic_quantized_model = qmanager.run_dynamic_quantization()

    # Save the dynamic quantized model
    qmanager.save_quantized_model(dynamic_quantized_model, config.get('dynamic_quantized_model_path', 'dynamic_quantized_model.pt'))

    # Run quantization-aware training (QAT)
    qat_model = qmanager.run_qat(num_epochs=config.get('qat_num_epochs', 10))

    # Save the QAT model
    qmanager.save_quantized_model(qat_model, config.get('qat_model_path', 'qat_model.pt'))

    # Load the quantized models
    loaded_static_model = qmanager.load_quantized_model(config.get('static_quantized_model_path', 'static_quantized_model.pt'))
    loaded_dynamic_model = qmanager.load_quantized_model(config.get('dynamic_quantized_model_path', 'dynamic_quantized_model.pt'))
    loaded_qat_model = qmanager.load_quantized_model(config.get('qat_model_path', 'qat_model.pt'))

    # Compare float and quantized models
    logger.info("\nComparing float and static quantized models:")
    qmanager.compare_models(model, loaded_static_model)

    logger.info("\nComparing float and dynamic quantized models:")
    qmanager.compare_models(model, loaded_dynamic_model)

    logger.info("\nComparing float and QAT models:")
    qmanager.compare_models(model, loaded_qat_model)

    # Validate accuracy
    logger.info("\nValidating float model:")
    float_accuracy = qmanager.validate_accuracy(model, val_loader)

    logger.info("\nValidating static quantized model:")
    static_quant_accuracy = qmanager.validate_accuracy(loaded_static_model, val_loader)

    logger.info("\nValidating dynamic quantized model:")
    dynamic_quant_accuracy = qmanager.validate_accuracy(loaded_dynamic_model, val_loader)

    logger.info("\nValidating QAT model:")
    qat_accuracy = qmanager.validate_accuracy(loaded_qat_model, val_loader)

    # Print final comparison
    logger.info("\nFinal Comparison:")
    logger.info(f"Float model accuracy: {float_accuracy*100:.2f}%")
    logger.info(f"Static quantized model accuracy: {static_quant_accuracy*100:.2f}%")
    logger.info(f"Dynamic quantized model accuracy: {dynamic_quant_accuracy*100:.2f}%")
    logger.info(f"QAT model accuracy: {qat_accuracy*100:.2f}%")

    # Calculate and log accuracy differences
    logger.info("\nAccuracy Differences:")
    logger.info(f"Static quantization accuracy change: {(static_quant_accuracy - float_accuracy)*100:.2f}%")
    logger.info(f"Dynamic quantization accuracy change: {(dynamic_quant_accuracy - float_accuracy)*100:.2f}%")
    logger.info(f"QAT accuracy change: {(qat_accuracy - float_accuracy)*100:.2f}%")

    # Optionally, you could add a threshold check here
    accuracy_threshold = config.get('accuracy_threshold', 0.02)  # 2% threshold
    if any(abs(acc - float_accuracy) > accuracy_threshold for acc in [static_quant_accuracy, dynamic_quant_accuracy, qat_accuracy]):
        logger.warning("One or more quantized models have accuracy significantly different from the float model.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise