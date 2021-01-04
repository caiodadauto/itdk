import logging
import logging.handlers as handlers


def create_logger(file_path):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_logger = logging.getLogger("Logger")
    file_logger.setLevel(logging.INFO)
    file_handler = handlers.RotatingFileHandler(
        file_path, maxBytes=200 * 1024 * 1024, backupCount=1
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_logger.addHandler(file_handler)
    return file_logger
