import logging

from .glossary import core
from .pretty_glossary import pretty_core

logger = logging.getLogger('glossaries')
if len(logger.handlers) == 0:  # ensure reload() doesn't add another handler
    logger.addHandler(logging.NullHandler())

