
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
import nltk

notstopwords = set(('not', 'no', 'mustn', "mustn\'t"))
stopwords = set(stopwords.words('english')) - notstopwords

lemmatizer = WordNetLemmatizer()
T = tokenizer.TweetTokenizer(preserve_handles=False, preserve_hashes=False, preserve_case=False, preserve_url=False, regularize=True)

def data_preprocessing(path_tweets):
	tweets = pd.read_csv(path_tweets, encoding='utf-8',sep=',')
	tweets['text'] = tweets['text'].apply(lambda x: standardization(x))
	tweets['sentiment'] = tweets['airline_sentiment'].apply(lambda x:0 if x=='negative' else (1 if x=='neutral' else 2))
	return tweets['text'], tweets['sentiment']



def data_preprocessing (path_tweets,corpora):
	data = pd.read_csv(path_tweets, encoding='utf-8',sep='\t', names=['id','class','text'])
	if corpora=='train':
		data['class'] = data['class'].apply(lambda x:0 if x=='negative' else (1 if x=='neutral' else 2 ))  # 0: 	negative, 1: neutral, 2: positive
	data['text'] = data['text'].apply(lambda x: standardization(x))
	return data['text'], data['class']


def standardization(tweet):
    tweet = re.sub(r"\\u2019", "'",tweet)
    tweet = re.sub(r"\\u002c", "'",tweet)
    tweet = re.sub(r" [0-9]+ "," ",tweet)
    tweets = T.tokenize(tweet)
    tweets = emoji.str2emoji(tweets)
    tweets = [lemmatizer.lemmatize(word,grammar[0].lower()) if grammar[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(word) for word,grammar in pos_tag(tweets)]
    tweets = [tweet for tweet in tweets if (tweet not in punctuation) and (tweet not in stopwords)]
    tweets = list(filter(lambda x: x.count('.') < 4, tweets))
    tweet = ' '.join(tweets)
    return tweet

def standardization2(tweet):
    tweet = re.sub(r"\\u2019", "'",tweet)
    tweet = re.sub(r"\\u002c", "'",tweet)
    tweet = re.sub(r" [0-9]+ "," ",tweet)
    tweet = re.sub(r"RT ", "", tweet)
    tweets = T.tokenize(tweet)
    tweets = emoji.str2emoji(tweets)
    tweets = [lemmatizer.lemmatize(word,grammar[0].lower()) if grammar[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(word) for word,grammar in pos_tag(tweets)]
    tweets = [tweet for tweet in tweets if (tweet not in punctuation) and (tweet not in stopwords)]
    tweets = list(filter(lambda x: x.count('.') < 4, tweets))
    return tweets

def create_dataset_word2Vec(tweet):
    return standardization2(tweet)

def read_dataset(path):
    df = pd.read_csv(path, sep="\t")
    df["Tweet"] = df["Tweet"].apply(lambda x: standardization(x))
    df["Intensity Class"] = df["Intensity Class"].apply(lambda x: x[:x.index(":")])
    return df["Tweet"], df["Intensity Class"]
