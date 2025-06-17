import os
import yaml
import torch

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve dynamic values
    if config.get("device", "auto") == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config
