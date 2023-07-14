import logging
import datetime
import random
import os


def create_logger(name="Training", path=None, extra_debug=None,
                  date_in_file=True, combined=True, unique_logger=True,
                  stream_lvl=None, file_lvl=None):
    if combined:
        file_name = "all"
    else:
        file_name = name

    if date_in_file:
        dt = datetime.datetime.now().strftime("%Y-%m-%d")
        file_name = f"{dt}-" + file_name
    file_name += ".log"

    if unique_logger:
        unique_name = str(random.random())  # Random unique
    else:
        unique_name = name

    logger = logging.getLogger(unique_name)
    logger.setLevel("DEBUG")

    # Log Handlers: Console and file
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", ""))
    # if path:
    #     log_dir = os.path.abspath(path)
    # else:
    #     log_dir = os.path.abspath('')

    # log_dir = os.path.join(log_dir, 'logs')

    try:
        fh = logging.FileHandler(os.path.join(log_dir, file_name),
                                 mode='a')
    except FileNotFoundError:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, file_name),
                                 mode='a')

    ch = logging.StreamHandler()

    # LEVEL
    if stream_lvl:
        ch.setLevel(stream_lvl)
    else:
        ch.setLevel("DEBUG")
    if file_lvl:
        fh.setLevel(file_lvl)
    else:
        fh.setLevel("DEBUG")

    # Log Formatting
    formatter = logging.Formatter(
            f'%(asctime)s -{name}- %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    if extra_debug:
        extra_fh = logging.FileHandler(os.path.join(log_dir, extra_debug),
                                       mode='a')
        extra_fh.setLevel("DEBUG")
        extra_fh.setFormatter(formatter)
        logger.addHandler(extra_fh)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = True  # this prevents other loggers I thinks from logging

    return logger


logger = create_logger()
