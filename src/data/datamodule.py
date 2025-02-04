import torch
import lightning as L
from pathlib import Path
from typing import Union, Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

class CatDogImageDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_dir: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 8,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        
        # Validate and store split ratios
        assert sum(train_val_test_split) == 1.0, "Split ratios must sum to 1"
        self.train_ratio, self.val_ratio, self.test_ratio = train_val_test_split
        
        self.dataset_path = self.data_dir / "cats_and_dogs_filtered"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download images if not already downloaded."""
        if not self.dataset_path.exists():
            download_and_extract_archive(
                url="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
                download_root=str(self.data_dir),
                remove_finished=True
            )

    def setup(self, stage: str = None):
        """Setup datasets with proper splits."""
        if self.train_dataset is not None:
            return  # Skip if already setup
            
        # Load the full dataset
        full_dataset = ImageFolder(
            root=self.dataset_path / "train",  # Use train folder for all splits
            transform=self.train_transform
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Update transforms for validation and test
        self.val_dataset.dataset.transform = self.valid_transform
        self.test_dataset.dataset.transform = self.valid_transform

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) 