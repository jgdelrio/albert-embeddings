from os import getenv
import pathlib


# API Info
API_VERSION = "0.0.1"

# Parameters
LOG_LEVEL = getenv("LOG_LEVEL", "INFO")
ROOT = pathlib.Path(__file__).parents[1]
PORT = getenv("PORT", "5000")

# Model Info & Parameters
DEFAULT_MODEL = getenv("DEFAULT_MODEL", "albert-base-v2")
MODELS = {
    'albert-base-v2': {
        "name": "ALBERT-Base",
        "version": "2.0",
        "reference": ROOT.joinpath('model', 'albert-base-v2').as_posix(),
        "hidden_size": 768,
    },
    'albert-large-v1': {
        "name": "ALBERT-Large",
        "version": "1.0",
        "reference": ROOT.joinpath('model', 'albert-large-v1').as_posix(),
        "hidden_size": 768,
    },
    'albert-xxlarge-v1': {
        "name": "ALBERT-xxLarge",
        "version": "1.0",
        "reference": ROOT.joinpath('model', 'albert-xxlarge-v1').as_posix(),
        "hidden_size": 768,
    },
}
