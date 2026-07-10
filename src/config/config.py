from pathlib import Path

import yaml

CONFIG_PATH = Path("src/config/config.yaml")

def load_config():
    with open(CONFIG_PATH,"r") as f:
        config = yaml.safe_load(f)

    return config