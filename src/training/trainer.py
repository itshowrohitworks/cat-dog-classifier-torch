import torch
import torch.nn as nn
import torch.optim as optim

from src.logger import logger
from src.models.cnn import CatDogCNN
from src.training.engine import train_one_epoch

def train_model(
  train_loader,
  epochs = 10,
  learning_rate = 0.001      
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Using Device: {device}")

    model = CatDogCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # trainig loop:
    for epoch in range(epochs):

        logger.info(f"Epoch: {epoch+1}/{epochs}")

        train_loss,train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {train_acc:.4f}"
        )
    
    return model