import pytest
from hydra.utils import instantiate
from src.data.datamodule import CatDogImageDataModule

def test_cat_dog_datamodule_instantiation(cfg):
    """Test if datamodule can be instantiated from config."""
    datamodule = instantiate(cfg.data)
    assert datamodule is not None

def test_cat_dog_datamodule_setup(cfg):
    """Test if datamodule setup works and creates correct splits."""
    datamodule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    
    # Check if datasets are created
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    
    # Check split ratios (approximately)
    total_size = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
    assert abs(len(datamodule.train_dataset) / total_size - 0.8) < 0.01
    assert abs(len(datamodule.val_dataset) / total_size - 0.1) < 0.01
    assert abs(len(datamodule.test_dataset) / total_size - 0.1) < 0.01

def test_cat_dog_datamodule_dataloaders(cfg):
    """Test if dataloaders work correctly."""
    datamodule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    # Check batch size
    batch = next(iter(train_loader))
    assert len(batch) == 2  # (images, labels)
    assert batch[0].shape[0] == cfg.data.batch_size  # batch size
    assert batch[0].shape[1] == 3  # channels
    assert batch[0].shape[2] == 96  # height
    assert batch[0].shape[3] == 96  # width 