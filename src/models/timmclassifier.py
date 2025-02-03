import lightning as L
import torch
import torch.nn.functional as F
import timm
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = 'mobilenetv2_100',
        num_classes: int = 2,
        pretrained: bool = True,
        lr: float = 1e-3,
        lr_scheduler_patience: int = 3,
        lr_scheduler_factor: float = 0.1,
        lr_scheduler_min_lr: float = 1e-6,
        optimizer_weight_decay: float = 1e-4,
        optimizer_beta1: float = 0.9,
        optimizer_beta2: float = 0.999,
        optimizer_eps: float = 1e-8,
    ):
        super().__init__()
        
        # Model parameters
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            global_pool='avg',
        )
        
        # Optimizer parameters
        self.lr = lr
        self.weight_decay = optimizer_weight_decay
        self.beta1 = optimizer_beta1
        self.beta2 = optimizer_beta2
        self.eps = optimizer_eps
        
        # Scheduler parameters
        self.scheduler_patience = lr_scheduler_patience
        self.scheduler_factor = lr_scheduler_factor
        self.scheduler_min_lr = lr_scheduler_min_lr

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_acc(preds, y)
        
        # Updated metric names with '/' separator
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_acc(preds, y)
        
        # Updated metric names with '/' separator
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.test_acc(preds, y)
        
        # Updated metric names with '/' separator
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", self.test_acc, prog_bar=True)
        return {"test/loss": loss, "test/acc": self.test_acc}

    def configure_optimizers(self):
        # Fixed optimizer type (Adam) with configurable parameters
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        
        # Fixed scheduler type (ReduceLROnPlateau) with configurable parameters
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.scheduler_min_lr,
                verbose=True
            ),
            "monitor": "val/loss",  # Updated metric name
            "interval": "epoch",
            "frequency": 1
        }
        
        return [optimizer], [scheduler] 