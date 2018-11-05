import logging

from .Extracter import Extractor
from .Categorizer import Categorizer
from .Tagger import Tagger
from .config import *

logger = logging.getLogger('matcher')
if len(logger.handlers) == 0:  # ensure reload() doesn't add another handler
    logger.addHandler(logging.NullHandler())

