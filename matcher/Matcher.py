import textacy
import nltk

import textacy.keyterms as tkt

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class Matcher:

    def __init__(self, text=None, url=None):
        """ Create instance of Matcher with text """

        self.keyphrases = []

        if url:
            self.parse(url)
        else:
            self.set_text(text)

    def set_text(self, text):
        """ Set text to the instance """

        self.text = text

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

        self.text = soup.get_text()

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
            self.text = ' '.join(filtered_sentence)

    def extract_keyphrases(self, algorithm, n=20, **kwargs):
        """ Method for extracting keyphrases from text
            algorithm takes 'str' object -> get function using eval
                           'func' object
            **kwargs: parameters for algorithm
        """

        if isinstance(algorithm, str):
            algorithm = eval('tkt.{}'.format(algorithm))

        doc = textacy.Doc(self.text, lang='en')
        self.keyphrases = list(algorithm(doc, **kwargs)[:n])

