import rootutils

# Set up root directory before any other imports
root = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True
)

import os
import sys
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from pathlib import Path

# Get project root directory
PROJECT_ROOT = str(Path(__file__).parent.parent.absolute())
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

# Add src to Python path
src_path = os.path.join(PROJECT_ROOT, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture(scope="session", autouse=True)
def project_root():
    """Change to project root for duration of tests."""
    original_dir = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield PROJECT_ROOT
    os.chdir(original_dir)

@pytest.fixture(scope="session")
def cfg() -> DictConfig:
    """Create a config for testing."""
    with initialize(version_base="1.3", config_path="../configs"):
        # Override the model and data module paths to use absolute imports
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=catdog_ex",
                "model._target_=src.models.timmclassifier.TimmClassifier",
                "data._target_=src.data.datamodule.CatDogImageDataModule",
            ]
        )
        return cfg

@pytest.fixture(scope="session")
def fast_dev_cfg(cfg: DictConfig) -> DictConfig:
    """Create a fast dev run config for testing."""
    # Add fast_dev_run to trainer config
    if "trainer" not in cfg:
        cfg.trainer = {}
    cfg.trainer.fast_dev_run = True
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 1
    
    # Update data config
    cfg.data.num_workers = 0
    cfg.data.batch_size = 2
    
    # Explicitly disable testing
    cfg.test = False
    
    return cfg 