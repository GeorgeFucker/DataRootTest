import pprint

from matcher import Tagger, DBInterface

tag = None

GET_ALL_TAGS_NAMES_CMD = 'SELECT * FROM tags;'

NGRAMS = (1, 2, 3)
WINDOW_WIDTH = 1200

TO_EXCLUDE = {'approach', 'application'}

if __name__ == '__main__':

    urls_tags = {
        'https://medium.com/s/story/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe': {},
        'https://blog.openai.com/language-unsupervised/': {}}

    db = DBInterface(dbname='postgres', user='postgres')
    tags = db.fetchall(GET_ALL_TAGS_NAMES_CMD)

    tagger = Tagger(tags=tags, to_exclude=TO_EXCLUDE, ngrams=NGRAMS, window_width=WINDOW_WIDTH)

    for url in urls_tags.keys():
        tagger.parse(url)
        urls_tags[url]['summary'] = tagger.summarize()
        urls_tags[url]['readtime'], urls_tags[url]['readability'] = tagger.estimate()
        urls_tags[url]['tags'] = tagger(url=url)

    pp = pprint.PrettyPrinter()
    pp.pprint(pp)
