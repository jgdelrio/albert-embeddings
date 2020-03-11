import re
import sys
import logging
import numpy as np
from albert_emb.config import LOG_LEVEL


has_point = re.compile(r'\.[\s*]$')


def eval_ending(text):
    text = text.strip()
    if text.endswith("..."):
        text += " ."
    elif not text.endswith("."):
        text += "."
    return text


def paragraphs_join(paragraphs):
    text = list(map(eval_ending, paragraphs))
    return " ".join(text)


def get_logger(name='albert-api', level=LOG_LEVEL):
    # Logger definition
    logger = logging.getLogger(name)
    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    logger.setLevel(levels[level])

    # ch = logging.StreamHandler(sys.stdout)
    # ch.setLevel(levels[level])
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def update_precision(emb, precision: str):
    if precision == 'float16':
        return np.float16(emb).tolist()
    elif precision == 'int8':
        return np.int8(pow(2, 8) * emb).tolist()
    elif precision == 'int16':
        return np.int16(pow(2, 16) * emb).tolist()
    elif precision == 'int32':
        return np.int32(pow(2, 32) * emb).tolist()
    else:
        return np.float32(emb).tolist()


logger = get_logger()
