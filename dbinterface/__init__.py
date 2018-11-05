import logging

from .DBInterface import DBInterface
from .consts import *

logger = logging.getLogger('dbinterface')
if len(logger.handlers) == 0:  # ensure reload() doesn't add another handler
    logger.addHandler(logging.NullHandler())

