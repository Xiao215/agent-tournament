import logging

DEFAULT_FORMATTER = logging.Formatter('%(message)s')

def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO
) -> logging.Logger:
    handler = logging.FileHandler(log_file)
    handler.setFormatter(DEFAULT_FORMATTER)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
