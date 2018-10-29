import tensorflow as tf
import pandas as pd
import re
from tensorflow import keras
import emoji
import preprocessing_git
from gensim.models import Word2Vec

def createWord2Vec(fileData, modelFile):
    data = []
    test_file = open('./test_file.txt', 'w')
    with open(fileData, 'r') as f:
        for line in f.readlines():
            data.append(preprocessing_git.create_dataset_word2Vec(line))
    
    # train model
    model = Word2Vec(data, size=300, window=3, min_count=1, sg=1)

    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words)

    # save model
    model.save(modelFile)

if __name__ == '__main__':
    l = createWord2Vec('./test_data.txt', 'model.bin')
    new_model = Word2Vec.load('model.bin')
