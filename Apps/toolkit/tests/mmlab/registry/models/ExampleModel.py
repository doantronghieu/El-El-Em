import add_packages
        
import pytest
import torch
import torch.nn as nn
from toolkit.ml.mmlab.engine.bases.model import MyBaseModel
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel

class ExampleModel(MyBaseModel):
    def __init__(self, num_classes=10, init_cfg=None, data_preprocessor=None):
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if self.data_preprocessor:
            inputs = self.data_preprocessor(inputs)
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if mode == 'loss':
            return {'loss': torch.mean(x)}
        elif mode == 'predict':
            return [{'pred': x[i]} for i in range(x.size(0))]
        else:
            return x

@pytest.fixture
def example_model():
    return ExampleModel(init_cfg={'type': 'Kaiming', 'layer': 'Conv2d'})

@pytest.fixture
def example_inputs():
    return torch.randn(2, 3, 32, 32)

def test_model_initialization(example_model: MyBaseModel):
    assert isinstance(example_model, MyBaseModel)

def test_forward_pass(example_model: MyBaseModel, example_inputs):
    outputs = example_model(example_inputs, mode='tensor')
    assert outputs.shape == (2, 10)

def test_loss_computation(example_model: MyBaseModel, example_inputs):
    loss_dict = example_model(example_inputs, mode='loss')
    assert 'loss' in loss_dict
    assert isinstance(loss_dict['loss'], torch.Tensor)

def test_prediction(example_model: MyBaseModel, example_inputs):
    predictions = example_model(example_inputs, mode='predict')
    assert len(predictions) == 2
    assert 'pred' in predictions[0]

def test_train_step(example_model: MyBaseModel, example_inputs):
    optim = SGD(example_model.parameters(), lr=0.01)
    optim_wrapper = OptimWrapper(optim)
    data = {'inputs': example_inputs, 'data_samples': None}
    log_vars = example_model.train_step(data, optim_wrapper)
    assert isinstance(log_vars, dict)

def test_val_and_test_step(example_model: MyBaseModel, example_inputs):
    data = {'inputs': example_inputs, 'data_samples': None}
    val_outputs = example_model.val_step(data)
    assert len(val_outputs) == 2
    assert 'pred' in val_outputs[0]

    test_outputs = example_model.test_step(data)
    assert len(test_outputs) == 2
    assert 'pred' in test_outputs[0]

def test_extract_feat(example_model: MyBaseModel, example_inputs):
    feats = example_model.extract_feat(example_inputs)
    assert feats.shape == (2, 10)

def test_add_model_specific_args():
    parser = ArgumentParser()
    parser = ExampleModel.add_model_specific_args(parser)
    args = parser.parse_args(['--custom_arg', 'test'])
    assert args.custom_arg == 'test'

# def test_ema(example_model: MyBaseModel):
#     example_model.to_ema()
#     assert hasattr(example_model, 'ema_model')
#     assert isinstance(example_model.ema_model, ExampleModel)
    
#     # Check that the ema_model has the same structure as the original model
#     for param_name, param in example_model.named_parameters():
#         assert hasattr(example_model.ema_model, param_name)
#         ema_param = getattr(example_model.ema_model, param_name)
#         assert param.shape == ema_param.shape

#     # Test updating EMA
#     original_params = [param.clone() for param in example_model.parameters()]
#     example_model.to_ema(momentum=0.9)
#     for orig_param, param, ema_param in zip(original_params, 
#                                             example_model.parameters(),
#                                             example_model.ema_model.parameters()):
#         assert torch.allclose(ema_param, 0.9 * orig_param + 0.1 * param, atol=1e-6)

# def test_distributed(example_model: MyBaseModel):
#     distributed_model = example_model.to_distributed()
#     assert isinstance(distributed_model, DistributedDataParallel)

def test_stack_batch(example_model: MyBaseModel):
    tensors = [torch.randn(3, 32, 32), torch.randn(3, 30, 30), torch.randn(3, 28, 28)]
    stacked = example_model.stack_batch(tensors)
    assert stacked.shape == (3, 3, 32, 32)

def test_sync_batchnorm(example_model: MyBaseModel):
    bn = nn.BatchNorm2d(64)
    sync_bn = example_model.convert_sync_batchnorm(bn)
    assert isinstance(sync_bn, nn.SyncBatchNorm)
    
    reverted_bn = example_model.revert_sync_batchnorm(sync_bn)
    assert isinstance(reverted_bn, nn.BatchNorm2d)

def test_bias_init_with_prob(example_model: MyBaseModel):
    bias_init = example_model.bias_init_with_prob(0.01)
    assert isinstance(bias_init, float)

def test_detect_anomalous_params(example_model: MyBaseModel):
    # Introduce an anomalous parameter
    example_model.fc.weight.data[0, 0] = float('inf')
    anomalous_params = example_model.detect_anomalous_params(example_model)
    assert 'fc.weight' in anomalous_params

# pytest ExampleModel.py