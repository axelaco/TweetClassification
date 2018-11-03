import tensorflow as tf
import pandas as pd
import re
import numpy as np
import json
import os
import sys
import pickle
from tensorflow import keras
import emoji
import preprocessing_git
from gensim.models import Word2Vec, KeyedVectors

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100) + 1
    r = '\r[%s%s]%d%%' % ("#" * rate_num, " " * (100 - rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()

def emoji_valence(word2vec):
    return


def prepareDataset(modelFile):
    data = []
    dir_file = '../tweet_data'
    files = os.listdir(dir_file)
    files = files[:5]
    for i in range(len(files)):
        with open(os.path.join(dir_file, files[i])) as f:
            lines = f.readlines()
            idx = 0
            for line in lines:
                view_bar(idx, len(lines))
                data.append(preprocessing_git.create_dataset_word2Vec(line))
                idx += 1
            print('\n')

    with open('data.bin', 'wb') as fp:
       pickle.dump(data, fp)

def smallDataset():
    dir_file = '../tweet_data'
    data = []
    with open(os.path.join(dir_file, 'second_debate_presidential.txt')) as f:
        lines = f.readlines()
        idx = 0
        for line in lines:
            view_bar(idx, len(lines))
            data.append(preprocessing_git.create_dataset_word2Vec(line))
            idx += 1
        print('\n')
    with open('data2.bin', 'wb') as fp:
       pickle.dump(data, fp)

    
def processEmojiValence(word):
    ones = np.ones(7)
    tmp = np.ones(7)
    json_data = json.load(open('../resources/emoji_valence.json'))
    for json_elt in json_data:
        if json_elt['emoji'] == word:
            tmp[json_elt['polarity'] + 3] = 0
            break
    return ones - tmp

def preprocessDepechMode():
    data = dict()
    f = open('../resources/depech_mood.txt', 'r')
    line = f.readline()
    ones = np.ones(4)
    PoS = {'a': 0, 'n': 1, 'v': 2, 'r': 3}
    for line in f.readlines():
        tmp = np.ones(4)
        arr = line.strip().split(';')
        firstCol = arr[0].split('#')
        word = firstCol[0]
        tmp[PoS[firstCol[1]]] = 0
        data[word] = np.append(ones - tmp, np.array(arr[1:], dtype=np.float))
    return data

def processDepechMode(data, word):
    return data[word]

def processEmojiSentimentLexicon(emoji):
    df = pd.read_csv('../resources/Emoji_Sentiment_Data_v1.0.csv')[['Emoji', 'Occurrences', 'Position', 'Negative', 'Neutral', 'Positive']]
    df.Occurrences = df.Occurrences.apply(lambda x: float(x))
    df.Negative = df.Negative.apply(lambda x: float(x))
    df.Neutral = df.Neutral.apply(lambda x: float(x))
    df.Positive = df.Positive.apply(lambda x: float(x))
    
    df.Negative = df.Negative / df.Occurrences
    df.Neutral = df.Neutral / df.Occurrences
    df.Positive = df.Positive / df.Occurrences
    elt = df[df['Emoji'] == emoji][['Position', 'Negative', 'Neutral', 'Positive']]
    if len(elt) > 0:
        return elt.values[0]
    else:
        return np.zeros(4)

def processOpinionLexiconEnglish(word):
    f_pos = open('../resources/opinion-positive-words.txt', 'r')
    f_neg = open('../resources/opinion-negative-words.txt', 'r')
    lines_pos = set(f_pos.read().splitlines())
    lines_neg = set(f_neg.read().splitlines())
    if (word in lines_pos):
        return np.array([1, 0])
    elif (word in lines_neg):
        return np.array([0, 1])
    else:
        return np.zeros(2)

def createWord2Vec(modelFile, dataset):
    data = pickle.load(open(dataset, 'rb'))
    # train model
    model = Word2Vec(data, size=300, window=3, min_count=1, sg=1)

    # summarize vocabulary
    words = list(model.wv.vocab)

    # save model
    model.save(modelFile)

def process_word2Vec(modelFile):
    model = Word2Vec.load(modelFile)
    test = np.array(model.wv.word_vec('men'))
    test = np.append(test, 0.3)
    print(test.shape)    


if __name__ == '__main__':
    #createWord2Vec('model2.bin', 'data2.bin')
    #process_word2Vec('model2.bin')
    print(processOpinionLexiconEnglish('abrasive'))
