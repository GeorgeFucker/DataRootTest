import bs4
from urllib.request import Request, urlopen
import requests
from textstat.textstat import textstat
# Words per minute
WPM = 266
WORD_LENGTH = 5
IMG_WEIGHT = 0.15

# extracting all text
def extract_text(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    soup = bs4.BeautifulSoup(webpage, 'html.parser')
    texts = soup.findAll(text=True)
    return texts

def is_visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif isinstance(element, bs4.element.Comment):
        return False
    elif element.string == "\n":
        return False
    return True

# filter all unimportant data
def filter_visible_text(page_texts):
    return filter(is_visible, page_texts)

def count_words_in_text_through_average_word_length(text_list, word_length):
    total_words = 0
    for current_text in text_list:
        total_words += len(current_text)/word_length
    return total_words

def get_img_cnt(url):
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content,'html.parser')
    return len(soup.find_all('img'))

# actual estimation
def estimate_reading_time_and_readability(url):
    texts = extract_text(url)
    filtered_text = filter_visible_text(texts)
    total_words = count_words_in_text_through_average_word_length(filtered_text, WORD_LENGTH)
    imgCounnt=get_img_cnt(url)
    text_readability=textstat.flesch_reading_ease(' '.join(list(filter_visible_text(texts))))
    if text_readability>70.0:
        readability_res= 'this is an easy to read article'
    elif text_readability<70.0 and text_readability>60.0:
        readability_res = 'this article is easy to read even for a beginner'
    elif text_readability<60.0 and text_readability>50.0:
        readability_res = 'this is an article for student with some knowledge'
    elif text_readability<50.0 and text_readability>30.0:
        readability_res = 'this is an article for student with strong knowledge of basics'
    elif text_readability<30.0:
        readability_res = 'this is a relatively hard to read article'
    return round((total_words/WPM)+imgCounnt*IMG_WEIGHT),readability_res

# example = estimate_reading_time_and_readability("https://towardsdatascience.com/an-introduction-to-bayesian-inference-in-pystan-c27078e58d53")
# print(example[1])
# print ('{} minutes to read'.format(example[0]))


