from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
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
LANGUAGE = "english"
SENTENCES_COUNT = 4


# or for plain text files
# parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
def get_summ(url,func=Summarizer1):
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = func(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)

#get_summ("https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1")
# get_summ(Summarizer2)
# get_summ(Summarizer3)
# get_summ(Summarizer4)
# get_summ(Summarizer5)
# get_summ(Summarizer6)
