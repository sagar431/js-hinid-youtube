import pytest
from hydra.utils import instantiate
import lightning as L

def test_train_model_fast_dev_run(fast_dev_cfg):
    """Test if training runs without errors in fast_dev_run mode."""
    # Initialize components
    model = instantiate(fast_dev_cfg.model)
    datamodule = instantiate(fast_dev_cfg.data)
    
    # Initialize trainer with fast_dev_run
    trainer = instantiate(
        fast_dev_cfg.trainer,
        callbacks=[],  # Disable callbacks for fast_dev_run
        logger=False,  # Disable logging for fast_dev_run
        _convert_="partial"
    )
    
    # Run training
    trainer.fit(model=model, datamodule=datamodule)
    
    # Check if training completed
    assert trainer.state.finished, "Training failed to complete"

def test_full_training_pipeline(fast_dev_cfg):
    """Test the complete training pipeline including callbacks and logging."""
    # Initialize components
    model = instantiate(fast_dev_cfg.model)
    datamodule = instantiate(fast_dev_cfg.data)
    
    # Initialize callbacks
    callbacks = []
    if "callbacks" in fast_dev_cfg:
        for _, cb_conf in fast_dev_cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(instantiate(cb_conf))
    
    # Initialize trainer
    trainer = instantiate(
        fast_dev_cfg.trainer,
        callbacks=callbacks,
        _convert_="partial"
    )
    
    # Run training and testing
    trainer.fit(model=model, datamodule=datamodule)
    results = trainer.test(model=model, datamodule=datamodule)
    
    # Check if we have test results
    assert results is not None
    assert len(results) > 0 