
"""
    ::param preserve_handles:: if False, strips Twitter user handles from text.
    ::param preverse_hashes:: if False, strips the hash symbol from hashtags (but does not delete the hashtag word)
    ::param preserve_cases:: if False, reduces text to lower case
    ::param preserve_url::  if False, strips url addresses from the text
    ::param regularize::  if True, regularizes the text for common English contractions, resulting in two word sequences
"""
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from tokenizer import tokenizer
import emoji
import re
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
import nltk
from unidecode import unidecode
tknzr = TweetTokenizer()


notstopwords = set(('not', 'no', 'mustn', "mustn\'t"))
stopwords = set(stopwords.words('english')) - notstopwords
text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'user'],
        
        fix_html=True,  # fix HTML tokens
        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter", 
        
        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter", 
        
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=True,  # spell correction for elongated words
        
        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize
)
lemmatizer = WordNetLemmatizer()
T = tokenizer.TweetTokenizer(preserve_handles=False, preserve_hashes=False, preserve_case=False, preserve_url=False, regularize=True)

def data_preprocessing(path_tweets):
	tweets = pd.read_csv(path_tweets, encoding='utf-8',sep=',')
	tweets['text'] = tweets['text'].apply(lambda x: standardization3(x))
	tweets['sentiment'] = tweets['airline_sentiment'].apply(lambda x:0 if x=='negative' else (1 if x=='neutral' else 2))
	return tweets['text'], tweets['sentiment']



def data_preprocessing(path_tweets,corpora):
	data = pd.read_csv(path_tweets, encoding='utf-8',sep='\t', names=['id','class','text'])
	if corpora=='train':
		data['class'] = data['class'].apply(lambda x:0 if x=='negative' else (1 if x=='neutral' else 2 ))
	data['text'] = data['text'].apply(lambda x: standardization3(x))
	return data['text'], data['class']


def data_preprocessing_test(path_tweets):
	data = pd.read_csv(path_tweets, encoding='utf-8',sep='\t')
	data['text'] = data['Tweet'].apply(lambda x: standardization3(x))
	return data['text']


def standardization(tweet):
	tweet = re.sub(r"\\u2019", "'", tweet)
	tweet = re.sub(r"\\u002c", "'", tweet)
	tweet=' '.join(emoji.str2emoji(unidecode(tweet).lower().split()))
	tweet = re.sub(r"(http|https)?:\/\/[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4}(/\S*)?", " ", tweet)
	tweet = re.sub(r"\'ve", " have", tweet)
	tweet = re.sub(r" can\'t", " cannot", tweet)
	tweet = re.sub(r"n\'t", " not", tweet)
	tweet = re.sub(r"\'re", " are", tweet)
	tweet = re.sub(r"\'d", " would", tweet)
	tweet = re.sub(r"\'ll", " will", tweet)
	tweet = re.sub(r"\'s", "", tweet)
	tweet = re.sub(r"\'n", "", tweet)
	tweet = re.sub(r"\'m", " am", tweet)
	tweet = re.sub(r"@\w+", r' ',tweet)
	tweet = re.sub(r"#\w+", r' ',tweet)
	tweet = re.sub(r" [0-9]+ "," ",tweet)
	tweet = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(i) for i,j in pos_tag(tknzr.tokenize(tweet))]
	tweet = [ i for i in tweet if (i not in stopwords) and (i not in punctuation ) ]
	tweet = ' '.join(tweet)
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
    tweet = ' '.join(tweets)
    return tweets


def standardization3(tweet):
    tweet=' '.join(emoji.str2emoji(tweet.split()))
    tweets = text_processor.pre_process_doc(tweet)
    tweets = emoji.str2emoji(tweets)
    tweets = [lemmatizer.lemmatize(word,grammar[0].lower()) if grammar[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(word) for word,grammar in pos_tag(tweets)]
    tweets = [tweet for tweet in tweets if (tweet not in punctuation) and (tweet not in stopwords)]
    #tweet = ' '.join(tweets)
    return tweets

def create_dataset_word2Vec(tweet):
    return standardization3(tweet)

def read_dataset(path):
    df = pd.read_csv(path, sep="\t")
    df["Tweet"] = df["Tweet"].apply(lambda x: standardization(x))
    df["Intensity Class"] = df["Intensity Class"].apply(lambda x: x[:x.index(":")])
    return df["Tweet"], df["Intensity Class"]


#print(standardization3("@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :‑D http://sentimentsymposium.com/."))
