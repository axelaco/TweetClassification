from tensorflow import keras as k
import pandas as pd
from unidecode import unidecode
import emoji
import re

def treatEmoji(tweet):
    return ' '.join(emoji.str2emoji(unidecode(tweet).lower().split()))


def tokenize(path):
    df = pd.read_csv(path, sep="\t")
    tweets = df["Tweet"]
    tweets = tweets.apply(lambda x : treatEmoji(x))
    t = k.preprocessing.text.Tokenizer()
    t.fit_on_texts(tweets)
    print(t.index_word)

def process_test():
    df = pd.read_csv('../resources/2018-Valence-oc-En-dev.txt', sep="\t")
    s = df['Intensity Class'].values
    data = []
    for i in s:
        try:
            data.append(re.search('[-0-9]+(?=:)', i).group(0))
        except:
            print(i)

    return data, df['Tweet'].values




if __name__ == '__main__':
    process_test()