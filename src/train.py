import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

from data.datamodule import CatDogImageDataModule
from models.classifier import CatDogClassifier
from utils import task_wrapper, setup_logger, logger

@task_wrapper
def train_model(data_module, model):
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="logs/catdog_classification/checkpoints",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    
    callbacks = [
        checkpoint_callback,
        RichProgressBar(),
        RichModelSummary(max_depth=1)
    ]

    # Initialize Trainer with minimal settings
    trainer = L.Trainer(
        max_epochs=5,
        callbacks=callbacks,
        accelerator="cpu",
        devices=1,
        logger=TensorBoardLogger(save_dir="logs", name="catdog_classification"),
        accumulate_grad_batches=8,  # Increase gradient accumulation
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
        precision="32",
        limit_train_batches=0.5,  # Use only 50% of training data
        limit_val_batches=0.5,    # Use only 50% of validation data
    )

    # Train and test the model
    logger.info("Starting model training")
    trainer.fit(model, data_module)
    logger.info(f"Best model path: {checkpoint_callback.best_model_path}")
    logger.info("Model training completed")
    
    logger.info("Starting model testing")
    test_result = trainer.test(model, data_module)
    logger.info(f"Test results: {test_result}")
    
    return trainer, test_result, checkpoint_callback.best_model_path

def main():
    # Setup logger
    setup_logger()
    logger.info("Starting main training script")

    # Set seed for reproducibility
    L.seed_everything(42, workers=True)
    logger.info("Set random seed to 42")

    # Initialize DataModule with minimal batch size
    data_module = CatDogImageDataModule(
        batch_size=4,  # Even smaller batch size
        num_workers=0
    )
    logger.info("Initialized DataModule")

    # Initialize Model
    model = CatDogClassifier(lr=1e-3)
    logger.info("Initialized Model")

    # Train and test
    trainer, test_result, best_model_path = train_model(data_module, model)
    
    # Save the best model path to a file for easy reference
    with open("best_model_path.txt", "w") as f:
        f.write(best_model_path)
    
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info("Training script completed successfully")

if __name__ == "__main__":
    main() 