import logging

from .Extracter import Extractor
from .DBInterface import DBInterface
from .Categorizer import Categorizer

logger = logging.getLogger('matcher')
if len(logger.handlers) == 0:  # ensure reload() doesn't add another handler
    logger.addHandler(logging.NullHandler())

