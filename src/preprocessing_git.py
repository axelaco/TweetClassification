
"""
    ::param preserve_handles:: if False, strips Twitter user handles from text.
    ::param preverse_hashes:: if False, strips the hash symbol from hashtags (but does not delete the hashtag word)
    ::param preserve_cases:: if False, reduces text to lower case
    ::param preserve_url::  if False, strips url addresses from the text
    ::param regularize::  if True, regularizes the text for common English contractions, resulting in two word sequences
"""
from tokenizer import tokenizer
import emoji
import re
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd


notstopwords = set(('not', 'no', 'mustn', "mustn\'t"))
stopwords = set( stopwords.words('english')) - notstopwords

lemmatizer = WordNetLemmatizer()
T = tokenizer.TweetTokenizer(preserve_handles=False, preserve_hashes=False, preserve_case=False, preserve_url=False, regularize=True)

def standardization(tweet):
    tweet = re.sub(r"\\u2019", "'",tweet)
    tweet = re.sub(r"\\u002c", "'",tweet)
    tweet = re.sub(r" [0-9]+ "," ",tweet)
    tweets = T.tokenize(tweet)
    tweets = emoji.str2emoji(tweets)
    tweets = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(i) for i,j in pos_tag(tweets)]
    tweets = [tweet for tweet in tweets if (tweet not in punctuation) and (tweet not in stopwords)]
    tweet = ' '.join(tweets)
    return tweet


def main():
    read_dataset('../resources/2018-Valence-oc-En-train.txt')

main()