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
from word2vecUtils import  processEmolex
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
    data = processEmolex()
    print(data['whimper'])