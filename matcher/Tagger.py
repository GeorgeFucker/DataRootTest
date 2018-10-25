import textacy
import nltk

import textacy.keyterms as tkt

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from textacy import similarity
from functools import reduce


class Tagger:
    """ Tagger class """

    def __init__(self, tags, to_proprocess=True, algorithm='sgrank', to_exclude=None, **kwargs):
        """ Initialize with given parameters """

        if tags:
            self._set_tags(tags)
        else:
            self.__tag_to_id = {}
            self.__id_to_tag = {}
            self.tags = set()

        self.__to_preprocess = to_proprocess
        self.__algorithm = algorithm
        self.__algorithm_params = kwargs

        self.text = None
        self.keyphrases = []
        self.matches = set()
        self.to_exclude = to_exclude

    def __call__(self, text=None, url=None, distance='hamming', threshold=0.7, no_rep=True, **kwargs):
        """ Create tags based on url or text """

        if not (text or url):
            return None

        # Parse text if given url
        self.text = self.parse(url) if url else text

        # Preprocess text if it should
        if self.__to_preprocess:
            self.text = self.preprocess(lowercase=True, no_punct=True, no_urls=True, no_stop_words=True)

        # Get keyphrases
        self.keyphrases = self.extract_keyphrases(algorithm=self.__algorithm, **self.__algorithm_params)

        # Find matches in our set of tags -> get set of text tags
        self.matches = self.match(distance=distance, threshold=threshold, no_rep=no_rep, **kwargs)

        return self.matches

    def parse(self, url):
        """ Get text from url """

        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
        except:
            print('Something wrong')

        soup = BeautifulSoup(webpage, 'html.parser')

        # Kill all scripts and style elements
        for script in soup(['script', 'style']):
            script.extract()

        return soup.get_text()

    def preprocess(self, lowercase=False, no_punct=False, no_urls=False, no_stop_words=False):
        """ Preprocess text for matching """

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in self.text.splitlines())

        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split(' '))

        # Drop blank lines
        self.text = '\n'.join(chunk for chunk in chunks if chunk)

        # Handle lowercase, no_pucnt and no_urls
        self.text = textacy.preprocess_text(self.text, lowercase=lowercase,
                                            no_punct=no_punct, no_urls=no_urls)

        # Handle stop words
        if no_stop_words:
            stop_words = set(stopwords.words('english'))
            word_tokens = nltk.word_tokenize(self.text)
            filtered_sentence = [w for w in word_tokens if w not in stop_words]
            return ' '.join(filtered_sentence)

    def extract_keyphrases(self, algorithm, **kwargs):
        """ Method for extracting keyphrases from text
            algorithm takes 'str' object -> get function using eval
                           'func' object
            **kwargs: parameters for algorithm
        """

        if isinstance(algorithm, str):
            algorithm = eval('tkt.{}'.format(algorithm))

        doc = textacy.Doc(self.text, lang='en')
        return list(algorithm(doc, **kwargs))

    def _set_tags(self, tags):
        """ Create dictionaries with tag_name : id and id : tag_name and tags
            'tags' is a list of tuples like ('id', 'tag_name')
        """

        self.__tag_to_id = {}
        self.__id_to_tag = {}
        self.tags = set()

        for id, tag in tags:
            self.__tag_to_id[tag] = id
            self.__id_to_tag[id] = tag
            self.tags.add(tag)

    def _distance(self, str1, str2, distance='hamming', **kwargs):
        """ Measure the similar between two strings using """

        if isinstance(distance, str):
            distance = eval('similarity.{}'.format(distance))

        return distance(str1, str2, **kwargs)

    def match(self, distance='hamming', threshold=0.7, no_rep=True, **kwargs):
        """ Match all tags that are similar
            **kwargs -> parameters for similarity function
        """

        matches = []

        for keyphrase in self.keyphrases:
            for tag in self.tags:
                if not tag:
                    continue
                d = self._distance(keyphrase[0], tag, distance=distance, **kwargs)
                if d > threshold:
                    matches.append((tag, d))

        if no_rep:
            matches_no_rep = set()
            for match in matches:
                match_set = set()
                match_set.add(match)

                for match_to_match in matches:
                    d = self._distance(match[0], match_to_match[0])
                    if d > threshold:
                        match_set.add(match_to_match)
                match = reduce(lambda x, y: x if x[1] > y[1] else y, match_set)
                matches_no_rep.add(match)

            matches = matches_no_rep

        if self.to_exclude:
            matches = {(match, p) for match, p in matches if match not in self.to_exclude}

        return matches
