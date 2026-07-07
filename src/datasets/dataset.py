from pathlib import Path
from torchvision import datasets
from src.logger import logger

def create_dataset(dir,transform=None):
    dataset = datasets.ImageFolder(
        root = Path(dir),
        transform = transform
    )

    logger.info(f"Dataset created with {len(dataset)} images from {dir}.")
    return dataset