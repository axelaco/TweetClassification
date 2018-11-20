
"""
    ::param preserve_handles:: if False, strips Twitter user handles from text.
    ::param preverse_hashes:: if False, strips the hash symbol from hashtags (but does not delete the hashtag word)
    ::param preserve_cases:: if False, reduces text to lower case
    ::param preserve_url::  if False, strips url addresses from the text
    ::param regularize::  if True, regularizes the text for common English contractions, resulting in two word sequences
"""
import tokenizer as tokenizer
import emoji as emoji
import re
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import nltk
import math

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download("stopwords")


notstopwords = set(('not', 'no', 'mustn', "mustn\'t"))
stopwords = set(stopwords.words('english')) - notstopwords
text_processor = TextPreProcessor(
    normalize=['url', 'email', 'user'],
    fix_html=True,  # fix HTML tokens
    segmenter="twitter", 
    corrector="twitter", 
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,
    tokenizer=SocialTokenizer(lowercase=True).tokenize)

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



# Same as preprocesing3 but with a join at the end
def preprocessing_two_class(tweet):
  tweet=' '.join(emoji.str2emoji(tweet.split()))
  tweets = text_processor.pre_process_doc(tweet)
  tweets = emoji.str2emoji(tweets)
  tweets = [lemmatizer.lemmatize(word,grammar[0].lower()) if
  grammar[0].lower() in ['a','n','v']  else
  lemmatizer.lemmatize(word) for word,grammar in pos_tag(tweets)]
  tweets = [tweet for tweet in tweets if (tweet not in
  punctuation) and (tweet not in stopwords)]
  tweet = ' '.join(tweets)
  return tweet

def create_dataset_word2Vec(tweet):
    new_tweet = standardization2(tweet)
    return new_tweet
    

def read_dataset(path):
    df = pd.read_csv(path, sep="\t")
    df["Tweet"] = df["Tweet"].apply(lambda x: standardization(x))
    df["Intensity Class"] = df["Intensity Class"].apply(lambda x: x[:x.index(":")])
    return df["Tweet"], df["Intensity Class"]
