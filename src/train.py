from pathlib import Path

from src.datasets.dataloader import create_dataloader
from src.logger import logger
from src.training.trainer import train_model

from src.config.config import load_config

def main():
    
    config = load_config()

    train_dir = Path(config["dataset"]["train_dir"])
    test_dir = Path(config["dataset"]["test_dir"])

    BATCH_SIZE = config["training"]["batch_size"]

    IMG_SIZE = config["dataset"]["image_size"]

    EPOCHES = config["training"]["epochs"]

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
        epochs=EPOCHES,
        learning_rate=config["training"]["learning_rate"]
    )

    logger.info("Training Completed Successfully!")

if __name__=="__main__":
    main()