import textacy
import nltk
import bs4
import requests

import textacy.keyterms as tkt

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from textacy import similarity
from functools import reduce
from textstat.textstat import textstat
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer1
# from sumy.summarizers.text_rank import TextRankSummarizer as Summarizer2
# from sumy.summarizers.luhn import LuhnSummarizer as Summarizer3
# from sumy.summarizers.lsa import LsaSummarizer as Summarizer4
# from sumy.summarizers.sum_basic import SumBasicSummarizer as Summarizer5
# from sumy.summarizers.kl import KLSummarizer as Summarizer6
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


class Tagger:
    """ Tagger class """

    # Words per minute
    WPM = 266
    WORD_LENGTH = 5
    IMG_WEIGHT = 0.15

    LANGUAGE = "english"
    SENTENCES_COUNT = 4

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

        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        soup = bs4.BeautifulSoup(webpage, 'html.parser')
        text = soup.findAll(text=True)
        return text

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

    ### Readtime and Readability estimation part
    def _is_visible(self, element):
        if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
            return False
        elif isinstance(element, bs4.element.Comment):
            return False
        elif element.string == "\n":
            return False
        return True

    # filter all unimportant data
    def _filter_visible_text(self, page_texts):
        return filter(self._is_visible, page_texts)

    def _count_words_in_text_through_average_word_length(self, text_list, word_length):
        total_words = 0
        for current_text in text_list:
            total_words += len(current_text) / word_length
        return total_words

    def _get_img_count(self, url):
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.content, 'html.parser')
        return len(soup.find_all('img'))

    # actual estimation
    def estimate_readtime_and_readability(self, url):
        filtered_text = self._filter_visible_text(self.text)
        total_words = self._count_words_in_text_through_average_word_length(filtered_text, self.WORD_LENGTH)
        img_count = self._get_img_count(url)
        text_readability = textstat.flesch_reading_ease(' '.join(list(self._filter_visible_text(self.text))))
        if text_readability > 70.0:
            readability_res = 'this is an easy to read article'
        elif 70.0 > text_readability > 60.0:
            readability_res = 'this article is easy to read even for a beginner'
        elif 60.0 > text_readability > 50.0:
            readability_res = 'this is an article for student with some knowledge'
        elif 50.0 > text_readability > 30.0:
            readability_res = 'this is an article for student with strong knowledge of basics'
        elif text_readability < 30.0:
            readability_res = 'this is a relatively hard to read article'
        return round((total_words / self.WPM) + img_count * self.IMG_WEIGHT), readability_res

    ### Summarizer part
    def summarize(self, url, func=Summarizer1):
        parser = HtmlParser.from_url(url, Tokenizer(self.LANGUAGE))
        stemmer = Stemmer(self.LANGUAGE)
        summarizer = func(stemmer)
        summarizer.stop_words = get_stop_words(self.LANGUAGE)
        for sentence in summarizer(parser.document, self.SENTENCES_COUNT):
            print(sentence)

