from os import getenv
import pathlib


# API Info
API_VERSION = "0.0.1"

# Model Info & Parameters
ALBERT = {
    "name": "ALBERT",
    "version": "1.0",
    "params": {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072
    }
}

# Parameters
LOG_LEVEL = getenv("LOG_LEVEL", "INFO")
ROOT = pathlib.Path(__file__).parents[1]
PORT = getenv("PORT", "5000")
