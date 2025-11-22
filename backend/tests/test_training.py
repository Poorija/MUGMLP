
import pytest
from backend.training import TASK_REGISTRY, SimpleNN
import torch

def test_task_registry_contains_auto_and_pytorch():
    assert "Auto" in TASK_REGISTRY["classification"]
    assert "Auto" in TASK_REGISTRY["regression"]
    assert "SimpleNN" in TASK_REGISTRY["classification"]
    assert "SimpleNN" in TASK_REGISTRY["regression"]

def test_simplenn_initialization():
    model = SimpleNN(input_dim=10, hidden_layers=[32, 16], output_dim=2, task_type="classification")
    assert isinstance(model, torch.nn.Module)

    # Check layers
    # input -> 32 -> relu -> 16 -> relu -> 2
    # Layers in sequential: Linear(10,32), ReLU, Linear(32,16), ReLU, Linear(16,2)
    # Total 5 modules
    assert len(model.model) == 5
