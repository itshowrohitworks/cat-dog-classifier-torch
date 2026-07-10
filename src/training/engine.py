import torch

from src.logger import logger

def train_one_epoch(
        model,
        dataloader,
        loss_fn,
        optimizer,
        device
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for img,labels in dataloader:
        img = img.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(img)

        loss = loss_fn(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _,predicted = torch.max(outputs,dim=1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = correct / total

    logger.info(
        f"Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_accuracy:.4f}"
    )

    return epoch_loss, epoch_accuracy