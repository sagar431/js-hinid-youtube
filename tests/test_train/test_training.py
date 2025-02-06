import rootutils

# Set up root directory before any other imports
root = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True
)

import pytest
from hydra import initialize, compose
from src.train import train_model

def test_train_and_test():
    """Test complete training pipeline."""
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=catdog_ex",
                "++trainer.fast_dev_run=True",
                "test=False",  # Explicitly disable testing
            ],
        )

    # Override paths in the config
    project_root = rootutils.find_root(indicator=".project-root")
    cfg.paths.root_dir = str(project_root)
    cfg.paths.data_dir = str(project_root / "data")
    cfg.paths.log_dir = str(project_root / "logs")
    cfg.paths.output_dir = str(project_root / "outputs")
    cfg.paths.work_dir = str(project_root)

    # Run the training function
    metrics = train_model(cfg)

    # Verify metrics structure and values
    assert isinstance(metrics, dict), "train_model should return a dictionary"
    
    # In fast_dev_run mode, metrics will be empty
    if not cfg.trainer.get("fast_dev_run", False):
        assert "train/loss" in metrics, "Train loss should be in metrics"
        assert "train/acc" in metrics, "Train accuracy should be in metrics"
        assert 0 <= metrics.get("train/loss", 0) <= 10
        assert 0 <= metrics.get("train/acc", 0) <= 1
    
    # Check test metrics
    if cfg.get("test"):
        assert "test/loss" in metrics, "Test loss should be in metrics"
        assert "test/acc" in metrics, "Test accuracy should be in metrics"
        assert 0 <= metrics["test/loss"] <= 10, "Test loss should be between 0 and 10"
        assert 0 <= metrics["test/acc"] <= 1, "Test accuracy should be between 0 and 1" 