from pathlib import Path
from torch.utils.data import DataLoader

from src.datasets.dataset import create_dataset
from src.datasets.transforms import (
    get_train_transform,
    get_test_transform
)

from src.logger import logger

def create_dataloader(
        train_dir,
        test_dir,
        batch_size=32,
        image_size=224,
        num_workers=0
):
    # Datasets:
    train_dataset = create_dataset(
        dir = train_dir,
        transform = get_train_transform(image_size) 
    )
    test_dataset = create_dataset(
        dir = test_dir,
        transform = get_test_transform(image_size)
    )

    # Dataloaders:
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )

    logger.info("Train and Test DataLoaders created.")

    return (
        train_loader,
        test_loader,
        train_dataset.classes
    )