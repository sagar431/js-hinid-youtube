import os
from typing import List, Tuple

import hydra
import lightning as L
import rootutils
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from hydra.utils import instantiate

# Project root setup
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.chdir(root)

from src.utils import task_wrapper, setup_logger, logger

def get_logger(cfg: DictConfig) -> List[Logger]:
    """Initialize loggers from config."""
    loggers: List[Logger] = []
    
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                logger.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(instantiate(lg_conf))
    
    return loggers

@task_wrapper
def train_model(cfg: DictConfig):
    """Train the model."""
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        logger.info(f"Set random seed to {cfg.seed}")

    # Initialize data module
    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = instantiate(cfg.data)

    # Initialize model
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate(cfg.model)

    # Initialize callbacks
    callbacks = []
    if "callbacks" in cfg:
        for cb_name, cb_conf in cfg.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))

    # Initialize loggers
    loggers = get_logger(cfg)

    # Initialize trainer
    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=loggers, 
        _convert_="partial"
    )

    # Train model
    logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Test model (only if not in fast_dev_run mode)
    if cfg.get("test") and not cfg.trainer.get("fast_dev_run", False):
        logger.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # Get metric score for hyperparameter optimization
    optimized_metric = trainer.callback_metrics.get(cfg.get("optimized_metric"))
    
    # Return empty dict if no metrics (fast_dev_run case)
    return {"train/loss": 0.0, "train/acc": 0.0} if optimized_metric is None else optimized_metric

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main training routine.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # Setup logger
    setup_logger()
    logger.info("Starting main training script")

    # Train model
    metric_value = train_model(cfg)

    # Logger info about metric value
    logger.info(f"Optimized metric value: {metric_value}")

if __name__ == "__main__":
    main() 