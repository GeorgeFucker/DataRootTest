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

import sumy.summarizers as sum
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
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

    def __init__(self, tags, to_preprocess=True, algorithm='sgrank', to_exclude=None, **kwargs):
        """ Initialize with given parameters """

        if tags:
            self._set_tags(tags)
        else:
            self.__tag_to_id = {}
            self.__id_to_tag = {}
            self.tags = set()

        self.__to_preprocess = to_preprocess
        self.__algorithm = algorithm
        self.__algorithm_params = kwargs

        self.html = None
        self.url = None
        self.text = None
        self.text_for_estimation = None
        self.img_count = 0
        self.keyphrases = []
        self.matches = set()
        self.to_exclude = to_exclude

    def __call__(self, text=None, url=None, html=None, distance='hamming',
                 threshold=0.7, no_rep=True, matches=None, **kwargs):
        """ Create tags based on url or text """

        if not (text or url or html):
            return None

        # Parse text if given url or html
        print('--> Parsing has started...')
        if url:
            self.parse(url=url)
        elif text:
            self.text = text
        elif html:
            self.parse(html=html)
        print('--> Parsing has done.')
        # Preprocess text if it should
        print('--> Preprocessing has started...')
        if self.__to_preprocess:
            self.preprocess(lowercase=True, no_punct=True, no_urls=True, no_stop_words=True)
        print('--> Preprocessing has done.')
        # Get keyphrases
        print('--> Extraction has started...')
        self.extract_keyphrases(algorithm=self.__algorithm, **self.__algorithm_params)
        print('--> Extraction has done.')
        # Find matches in our set of tags -> get set of text tags
        print('--> Matching has started...')
        self.match(distance=distance, threshold=threshold, no_rep=no_rep, matches=matches, **kwargs)
        print('--> Matching has done.')

        return self.matches

    def parse(self, url=None, html=None):
        """ Get text from url """

        if url:
            try:
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                html = urlopen(req).read()
            except:
                print('Something wrong')

        self.html = html
        soup = BeautifulSoup(html, 'html.parser')
        # Kill all scripts and style elements
        for script in soup(['script', 'style']):
            script.extract()

        self.text_for_estimation = soup.findAll(text=True)

        self.url = url
        self.text = soup.get_text()
        self.img_count = len(soup.find_all('img'))

    def _delete_stop_words(self, text):
        """ Delete stop words """

        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if w not in stop_words]
        return ' '.join(filtered_sentence)

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
            self.text = self._delete_stop_words(self.text)

    def extract_keyphrases(self, algorithm, **kwargs):
        """ Method for extracting keyphrases from text
            algorithm takes 'str' object -> get function using eval
                           'func' object
            **kwargs: parameters for algorithm
        """

        if isinstance(algorithm, str):
            algorithm = eval('tkt.{}'.format(algorithm))

        doc = textacy.Doc(self.text, lang='en')
        self.keyphrases = list(algorithm(doc, **kwargs))

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

    def _delete_repetitions(self, matches, threshold=0.9):
        """ Delete repetitions in matches """

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

        return matches_no_rep

    def _exclude_words(self, matches):
        """ Delete words that was pointed to delete """

        return {(match, p) for match, p in matches if match not in self.to_exclude}

    def match(self, distance='hamming', threshold=0.7, no_rep=True, matches=None, **kwargs):
        """ Match all tags that are similar
            **kwargs -> parameters for similarity function
        """

        if matches:
            matches = [(match, 1) for match in matches]
        else:
            matches = []

        while threshold >= 0:
            for keyphrase in self.keyphrases:
                for tag in self.tags:
                    if not tag:
                        continue
                    d = self._distance(keyphrase[0], tag, distance=distance, **kwargs)
                    if d > threshold:
                        matches.append((tag, d))
            if matches:
                print('All matches found')
                break

            threshold -= 0.1

        if no_rep:
            matches = self._delete_repetitions(matches)

        if self.to_exclude:
            matches = self._exclude_words(matches)

        self.matches = matches

    # Readtime and Readability estimation part
    def _is_visible(self, element):
        """ Return visibility of the element """

        if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
            return False
        elif isinstance(element, bs4.element.Comment):
            return False
        elif element.string == "\n":
            return False
        return True

    def _filter_visible_text(self):
        """ Filter all unimportant data """

        return filter(self._is_visible, self.text_for_estimation)

    def _count_words(self, text_list, word_length):
        """ Count word numbers int text"""

        total_words = 0
        for current_text in text_list:
            total_words += len(current_text) / word_length
        return total_words

    def estimate(self):
        """ Estimate readtime and readability"""

        filtered_text = self._filter_visible_text()
        total_words = self._count_words(filtered_text, self.WORD_LENGTH)
        text_readability = textstat.flesch_reading_ease(' '.join(list(self._filter_visible_text())))
        if text_readability > 70.0:
            readability_res = 'this is an easy to read article'
        elif 70.0 > text_readability > 60.0:
            readability_res = 'this article is easy to read even for a beginner'
        elif 60.0 > text_readability > 40.0:
            readability_res = 'this is an article for student with some knowledge'
        elif 40.0 > text_readability > 20.0:
            readability_res = 'this is an article for student with strong knowledge of basics'
        elif text_readability < 20.0:
            readability_res = 'this is a relatively hard to read article'
        return round((total_words / self.WPM) + self.img_count * self.IMG_WEIGHT), readability_res

    def _check_method(self, method):
        """ Check if the method is str or func """

        if isinstance(method, str):
            method_name = method.split(sep='_')
            method_name = list(map(lambda x: x[0].upper() + x[1:], method_name))
            method_name = ''.join(method_name)
            method_name += 'Summarizer'
            return eval('sum.{}.{}'.format(method, method_name))

    def summarize(self, method='luhn'):
        """ Summarize text """

        method = self._check_method(method)

        if self.url:
            parser = HtmlParser.from_url(self.url, Tokenizer(self.LANGUAGE))
        elif self.html:
            parser = HtmlParser(self.html, Tokenizer(self.LANGUAGE))
        stemmer = Stemmer(self.LANGUAGE)
        summarizer = method(stemmer)
        summarizer.stop_words = get_stop_words(self.LANGUAGE)
        sumy = summarizer(parser.document, self.SENTENCES_COUNT)
        summary = ''.join([str(i) for i in list(sumy)])

        return summary

