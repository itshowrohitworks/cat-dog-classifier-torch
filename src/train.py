from pathlib import Path

from src.datasets.dataloader import create_dataloader
from src.logger import logger
from src.training.trainer import train_model

def main():

    train_dir = Path("data/train")
    test_dir = Path("data/test")
    
    BATCH_SIZE = 32
    IMG_SIZE = 224
    EPOCHES = 10

    train_loader,test_loader,classes = create_dataloader(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    logger.info(f"Classes: {classes}")

    # Model:
    model = train_model(
        train_loader=train_loader,
        epochs=EPOCHES
    )

    logger.info("Training Completed Successfully!")

if __name__=="__main__":
    main()