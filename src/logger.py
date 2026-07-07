import logging
from pathlib import Path

curr_file = Path(__file__).resolve()

root = curr_file.parent.parent

LOG_DIR = root / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "running_logs.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="a"
)

logger = logging.getLogger(__name__)