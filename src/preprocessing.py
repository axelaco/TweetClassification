from tensorflow import keras as k
import pandas as pd
from unidecode import unidecode
import emoji


def treatEmoji(tweet):
    return ' '.join(emoji.str2emoji(unidecode(tweet).lower().split()))


def tokenize(path):
    df = pd.read_csv(path, sep="\t")
    tweets = df["Tweet"]
    tweets = tweets.apply(lambda x : treatEmoji(x))
    t = k.preprocessing.text.Tokenizer()
    t.fit_on_texts(tweets)
    print(t.index_word)