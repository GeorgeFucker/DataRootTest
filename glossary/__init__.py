import logging

from .Glossary import Glossary

logger = logging.getLogger('glossary')
if len(logger.handlers) == 0:  # ensure reload() doesn't add another handler
    logger.addHandler(logging.NullHandler())

