import rootutils
root = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True
)

import pytest
import torch
from hydra.utils import instantiate
from src.models.timmclassifier import TimmClassifier

def test_timm_classifier_instantiation(cfg):
    """Test if model can be instantiated from config."""
    model = instantiate(cfg.model)
    assert model is not None

def test_timm_classifier_forward(cfg):
    """Test if model forward pass works."""
    model = instantiate(cfg.model)
    batch_size = 2
    
    # Create dummy input
    x = torch.randn(batch_size, 3, 96, 96)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, cfg.model.num_classes)
    assert not torch.isnan(output).any()

def test_timm_classifier_training_step(cfg):
    """Test if training step works."""
    model = instantiate(cfg.model)
    batch_size = 2
    
    # Create dummy batch
    x = torch.randn(batch_size, 3, 96, 96)
    y = torch.randint(0, 2, (batch_size,))
    
    # Run training step
    loss = model.training_step((x, y), 0)
    
    # Check if loss is valid
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)

def test_timm_classifier():
    """Test complete model functionality."""
    model = TimmClassifier(model_name="resnet18", num_classes=2)

    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 96, 96)  # Using 96x96 as per our config
    output = model(input_tensor)

    # Check output shape and values
    assert output.shape == (batch_size, 2), "Output shape should be [batch_size, num_classes]"
    assert not torch.isnan(output).any(), "Output should not contain NaN values"

    # Test training step
    labels = torch.randint(0, 2, (batch_size,))
    batch = (input_tensor, labels)
    loss = model.training_step(batch, 0)

    # Check loss properties
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.shape == (), "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN" 