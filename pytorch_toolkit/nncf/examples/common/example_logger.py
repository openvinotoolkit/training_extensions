import logging
import sys

logger = logging.getLogger("example")
_LOGGER_INITIALIZED = False

if not _LOGGER_INITIALIZED:
    logger.setLevel(logging.INFO)
    hdl = logging.StreamHandler(stream=sys.stdout)
    hdl.setFormatter(logging.Formatter("%(message)s"))
    hdl.setLevel(logging.INFO)
    logger.addHandler(hdl)
