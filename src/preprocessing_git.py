from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
import emoji
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
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


def data_preprocessing(path_tweets,corpora):
	data = pd.read_csv(path_tweets, encoding='utf-8',sep='\t', names=['id','class','text'])
	if corpora=='train':
		data['class'] = data['class'].apply(lambda x:0 if x=='negative' else (1 if x=='neutral' else 2 ))
	data['text'] = data['text'].apply(lambda x: standardization(x))
	return data['text'], data['class']


def standardization(tweet):
    tweet=' '.join(emoji.str2emoji(tweet.split()))
    tweets = text_processor.pre_process_doc(tweet)
    tweets = emoji.str2emoji(tweets)
    tweets = [lemmatizer.lemmatize(word,grammar[0].lower()) if grammar[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(word) for word,grammar in pos_tag(tweets)]
    tweets = [tweet for tweet in tweets if (tweet not in punctuation) and (tweet not in stopwords)]
    return tweets


def create_dataset_word2Vec(tweet):
    return standardization(tweet)


def read_dataset(path):
    df = pd.read_csv(path, sep="\t")
    df["Tweet"] = df["Tweet"].apply(lambda x: standardization(x))
    df["Intensity Class"] = df["Intensity Class"].apply(lambda x: x[:x.index(":")])
    return df["Tweet"], df["Intensity Class"]