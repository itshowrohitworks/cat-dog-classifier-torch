from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


LOG_DIR = Path("runs")


def get_writer():

    LOG_DIR.mkdir(exist_ok=True)

    writer = SummaryWriter(LOG_DIR)

    return writer