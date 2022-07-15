import logging
import os
import sys

_module_logger: logging.Logger = None


# setup logging
def setup_logging():
    global _module_logger
    os.makedirs('./logs', exist_ok=True)
    if _module_logger is not None:
        return _module_logger
    loglevel = 'DEBUG'
    # logger
    _module_logger = logging.getLogger('tspd_osm')
    _module_logger.setLevel(logging.DEBUG)
    _module_logger.propagate = False

    # create formatters
    simple_formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    detailed_formatter = logging.Formatter("%(asctime)s %(name)s[%(process)d]: %(levelname)s - %(message)s")

    # create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, loglevel))
    console_handler.setFormatter(simple_formatter)

    # create file handler

    file_handler = logging.FileHandler('./logs/tspd-osm.debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # add handlers
    _module_logger.addHandler(console_handler)
    _module_logger.addHandler(file_handler)


setup_logging()



def get_logger() -> logging.Logger:
    return _module_logger

