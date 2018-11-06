from .config import LIMIT, STATUS, OMIT

GET_TAGS_CMD = 'SELECT * FROM public.tags;'
GET_ARTICLES_CMD = 'SELECT * FROM public."Articles" ' \
                   'WHERE articles_status != \'{}\' AND articles_source_id != \'{}\' ' \
                   'LIMIT {};'.format(STATUS, OMIT, LIMIT)

