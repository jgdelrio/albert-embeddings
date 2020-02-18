import re
import sys
import logging
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

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(levels[level])
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger()
