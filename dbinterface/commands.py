from .config import LIMIT

GET_TAGS_CMD = 'SELECT * FROM public.tags;'
GET_ARTICLES_CMD = 'SELECT * FROM public."Articles" ' \
                   'LIMIT {};'.format(LIMIT)

