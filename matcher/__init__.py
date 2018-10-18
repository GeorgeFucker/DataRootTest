import logging

from .Matcher import Matcher
from .DBInterface import DBInterface

logger = logging.getLogger('matcher')
if len(logger.handlers) == 0:  # ensure reload() doesn't add another handler
    logger.addHandler(logging.NullHandler())

