from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import emoji
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
import re
tknzr = TweetTokenizer()


notstopwords = set(('not', 'no', 'mustn', "mustn\'t"))
stopwords = set(stopwords.words('english')) - notstopwords



text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
        
        fix_html=True,  # fix HTML tokens
        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter", 
        
        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter", 
        
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct=True,  # spell correction for elongated words
        pell_correct_elong=True,
        
        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize
)
lemmatizer = WordNetLemmatizer()



emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}
emotion2label_teacher = {"others":3, "happy":1, "sad":2, "angry":0}

def data_preprocessing(path_tweets,corpora):
	data = pd.read_csv(path_tweets, encoding='utf-8',sep='\t', names=['id','class','text'])
	if corpora=='train':
		data['class'] = data['class'].apply(lambda x:0 if x=='negative' else (1 if x=='neutral' else 2 ))
	data['text'] = data['text'].apply(lambda x: standardization(x))
	return data['text'], data['class']

def data_preprocessing_semeval(path_data, corpora):
        data = pd.read_csv(path_data, encoding='utf-8', sep='\t')
        data['turn1'] = data['turn1'].apply(lambda x: standardization(x))
        data['turn2'] = data['turn2'].apply(lambda x: standardization(x))
        data['turn3'] = data['turn3'].apply(lambda x: standardization(x))
        if corpora=='train':
                data['label'] = data['label'].apply(lambda x: emotion2label[x])
                return data['id'], data['turn1'], data['turn2'], data['turn3'], data['label']
        return data['id'], data['turn1'], data['turn2'], data['turn3']

def load_data_semeval(path_tweet, label):
	try:
		if label=='True':
			tweet = pd.read_csv(path_tweet, encoding='utf-8',sep='\t')
			text = tweet['turn1']+" "+tweet['turn2']+" "+tweet['turn3']
			return text, tweet['label']
		else:
			tweet = pd.read_csv(path_tweet, encoding='utf-8',sep='\t')
			text = tweet['turn1']+" "+tweet['turn2']+" "+tweet['turn3']
			return text			
	except IOError: 
		print ("Could not read file:", path_tweet)


def data_preprocessing_teacher(path_tweet, label):
	if label=='True':
		texts, labels = load_data_semeval(path_tweet,label)
		texts = texts.apply(lambda x: standardization_teacher(x))
		labels = labels.apply(lambda x: emotion2label_teacher[x])	
		return texts, labels
	else:
		texts = load_data_semeval(path_tweet,label)
		texts = texts.apply(lambda x: standardization_teacher(x))	
		return texts

def standardization_teacher(tweet):
	tweet = re.sub(r"\\u2019", "'", tweet)
	tweet = re.sub(r"\\u002c", ",", tweet)
	tweet = tweet.lower()
	tweet=emoji.str2emoji(tweet)
	tweet = re.sub(r"(http|https)?:\/\/[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4}(/\S*)?", " ", tweet)
	tweet = re.sub(r"u r "," you are ",tweet)
	tweet = re.sub(r"U r "," you are ",tweet)
	tweet = re.sub(r" u(\s|$)"," you ",tweet)
	tweet = re.sub(r"didnt","did not",tweet)
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
	tweet = re.sub(r" plz[\s|$]", " please ",tweet)
	tweet = re.sub(r"^([1-9] |1[0-9]| 2[0-9]|3[0-1])(.|-)([1-9] |1[0-2])(.|-|)20[0-9][0-9]"," ",tweet)
	tweet = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(i) for i,j in pos_tag(tknzr.tokenize(tweet))]
	tweet = [ i for i in tweet if (i not in punctuation ) ]
	tweet = ' '.join(tweet)
	return tweet.lower()


def standardization(tweet):
    tweet=' '.join(emoji.str2emoji(tweet.split()))
    tweets = text_processor.pre_process_doc(tweet)
    tweets = [lemmatizer.lemmatize(word,grammar[0].lower()) if grammar[0].lower() in ['a','n','v']  else lemmatizer.lemmatize(word) for word,grammar in pos_tag(tweets)]
    tweets = [tweet for tweet in tweets if (tweet not in punctuation)]
    return tweets


def create_dataset_word2Vec(tweet):
    return standardization(tweet)


def read_dataset(path):
    df = pd.read_csv(path, sep="\t")
    df["Tweet"] = df["Tweet"].apply(lambda x: standardization(x))
    df["Intensity Class"] = df["Intensity Class"].apply(lambda x: x[:x.index(":")])
    return df["Tweet"], df["Intensity Class"]
