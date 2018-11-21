import numpy as np
import keras
from preprocessing_git import data_preprocessing
from gensim.models import KeyedVectors
from keras.utils.np_utils import to_categorical
import pickle
import pandas as pd
from scipy.stats import pearsonr
import os
import word2vecUtils
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Activation, Bidirectional,  Flatten, Input, GRU, GaussianNoise
from keras.models import load_model
from kutilities.layers import Attention
from keras.optimizers import Adam
import re
# DEBUG purpose
#import importlib
#importlib.reload(word2vecUtils)
from word2vecUtils import afin, emojiValence, depechMood, emolex, \
  emojiSentimentLexicon, opinionLexiconEnglish
from sklearn.model_selection import StratifiedKFold


word2vec_tweet = pickle.load(open('../resources/datastories.twitter.300d.pickle', 'rb'))
EMBEDDING_DIM = 344



# Word2Vec to KeyedVectors 
def saveKeyedVectors(path, model):
   model_vectors = model.wv 
   model_vectors.save(path)

def loadKeyedVectors(path):
   return KeyedVectors.load(path, mmap='r')



def get_max_len(tweets, tokenizer):
  return len(max([max(tokenizer.texts_to_sequences(tweet), key=len) for tweet in tweets], key=len))

def prepareData(corpora3, corpora7):
    tweet3, sentiment3 = data_preprocessing(corpora3, 'train')
    tweet7, sentiment7 = data_preprocessing(corpora7, 'test')

    all_tweet = tweet3.append(tweet7)

    tokenizer = keras.preprocessing.text.Tokenizer(filters=' ')
    tokenizer.fit_on_texts(all_tweet)
    word_index = tokenizer.word_index

    return word_index, tokenizer, tweet3, tweet7, sentiment3, sentiment7


def get_dataset(tweet, sentiment, max_len, tokenizer):
    sequences_train = tokenizer.texts_to_sequences(tweet)
    data_train = keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_len)
    indices_train = np.arange(data_train.shape[0])
    data_train = data_train[indices_train]
    labels_train = to_categorical(np.asarray(sentiment), 7)
    labels_train = labels_train[indices_train]

    return data_train, labels_train


def get_train_test(tweet, sentiment, max_len, tokenizer):
    sequences_train = tokenizer.texts_to_sequences(tweet)

    # print(np.asarray(sentiment).shape)

    data_train = keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_len)
    indices_train = np.arange(data_train.shape[0])
    data_train = data_train[indices_train]

    labels_train = to_categorical(np.asarray(sentiment), 7)
    labels_train = labels_train[indices_train]

    split_idx = int(len(data_train) * 0.80)
    x_train, x_val = data_train[:split_idx], data_train[split_idx:]
    y_train, y_val = labels_train[:split_idx], labels_train[split_idx:]

    return x_train, x_val, y_train, y_val



def concatenateEmbeding(word, wv_sentiment_dict):
  concat = [word2vec_tweet[word]]
  for keys in wv_sentiment_dict:
    dictionary, fct  = wv_sentiment_dict[keys]
    concat.append(fct(dictionary, word))

  return np.concatenate(concat)

def createEmbedingMatrix(word_index, w2vpath, dim):
    #word2vec = loadKeyedVectors(w2vpath)
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    oov = []
    oov.append((np.random.rand(dim) * 2.0) - 1.0)
    oov = oov / np.linalg.norm(oov)

    path = "../resources/embeding"


    # Load sentiment vectors
    sentiment_wv_dict = {
      'afin': [pickle.load(open(path + '/afin', 'rb')), afin],
      'ev': [pickle.load(open(path + '/EV', 'rb')), emojiValence],
      'depech': [pickle.load(open(path + '/depech', 'rb')), depechMood],
      'emolex': [pickle.load(open(path + '/emolex', 'rb')), emolex],
      'emoji': [pickle.load(open(path + '/EmojiSentimentLexicon', 'rb')),emojiSentimentLexicon],
      'opinion': [pickle.load(open(path + '/OpinionLexicon',
      'rb')),opinionLexiconEnglish]
    }


    for word, i in word_index.items():
        if word in word2vec_tweet:
            embedding_matrix[i] = concatenateEmbeding(word, sentiment_wv_dict)
        else:
            embedding_matrix[i] = oov

    return embedding_matrix


def model(x_train_3, y_train_3,x_val_3, y_val_3, embedding_layer):
    model2 = Sequential()
    model2.add(embedding_layer)
    model2.add(GaussianNoise(0.3))
    model2.add(Dropout(0.3))
    model2.add(Bidirectional(LSTM(150, recurrent_dropout=0.3, kernel_regularizer=l2(0), return_sequences=True)))
    model2.add(Dropout(0.3))
    model2.add(Bidirectional(LSTM(150, recurrent_dropout=0.3, kernel_regularizer=l2(0), return_sequences=True)))
    model2.add(Dropout(0.3))
    model2.add(Attention())
    model2.add(Dense(3, activity_regularizer=l2(0.0001)))
    model2.add(Activation('softmax'))
    model2.compile(optimizer=Adam(clipnorm=1, lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model2.summary()
    history=model2.fit(x_train_3, y_train_3, validation_data=(x_val_3, y_val_3),epochs=12, batch_size=50)
    model2.save("./model2.h5")

def model_final(modelPath, x_train_7, y_train_7, x_val_7, y_val_7):
  model = load_model(modelPath, custom_objects={"Attention": Attention})
  model.summary()
  model.layers.pop()
  model.layers.pop()
  model.add(Dense(150,activation='relu',name='dense1'))
  model.add(Dense(7,activation='softmax',name='dense2'))
  model.summary()
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
  history = model.fit(x_train_7, y_train_7,   validation_data=(x_val_7,y_val_7), epochs=18, batch_size=64)
  return model
  # model.save("./model7.h5")


def pearson_score(model, x_val_7, y_val_7):
    pred = model.predict(x_val_7)
    y_pred = np.argmax(pred, axis=1)
    score = pearsonr(np.argmax(y_val_7, axis=1), y_pred)[0]
    print(score)
    return score

def process_test():
    df = pd.read_csv('../resources/2018-Valence-oc-En-dev.txt', sep="\t")
    s = df['Intensity Class'].values
    data = []
    for i in s:
        data.append(int(re.search('[-0-9]+(?=:)', i).group(0)))

    return df['Tweet'].values, data


if __name__ == '__main__':
  corpora_train_3 = '../resources/data_train_3.csv'
  corpora_train_7 = '../resources/data_train_7.csv'
  corpora_test_7 = "'../resources/data_test_7.csv'"


  word_index, tokenizer, tweet3, tweet7, sentiment3, sentiment7 = prepareData(corpora_train_3, corpora_train_7)
  #model = Word2Vec.load('../resources/model_5M.bin')
  #saveKeyedVectors('../resources/model2.kv', model)
  sentiment7 = [x + 3 for x in sentiment7]
  
  MAX_SEQUENCE_LENGTH = get_max_len([tweet3, tweet7], tokenizer)

  embedding_matrix = createEmbedingMatrix(word_index, '../resources/model2.kv', EMBEDDING_DIM)

  x_train_7, x_val_7, y_train_7, y_val_7 = get_train_test(tweet7, sentiment7, MAX_SEQUENCE_LENGTH, tokenizer)

  # x_dataset, y_dataset = get_dataset(tweet3, sentiment3, MAX_SEQUENCE_LENGTH, tokenizer)
  x_dataset, y_dataset = get_dataset(tweet7, sentiment7, MAX_SEQUENCE_LENGTH, tokenizer)

  x_data, y_data = process_test()

  y_data = [x + 3 for x in y_data]

  x_val, y_val = get_dataset(x_data, y_data, MAX_SEQUENCE_LENGTH, tokenizer)

  # m = load_model('./model65.h5', custom_objects={"Attention": Attention})
  # pearson_score(m, x_dataset, y_dataset)
  # pearson_score(m, x_val, y_val)

  embedding_layer = Embedding(len(word_index) + 1,
                          EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          mask_zero=True,
                          trainable=False, name='embedding_layer')

  skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

  max_acc = 0

  for train_index, test_index in skf.split(x_dataset, np.argmax(y_dataset, axis=-1)):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = x_dataset[train_index], x_dataset[test_index]
    y_train, y_test = y_dataset[train_index], y_dataset[test_index]

    # model(x_train, y_train, x_test , y_test, embedding_layer)
    m = model_final("../resources/model2.h5", x_train, y_train, x_test, y_test)
    acc = pearson_score(m, x_val, y_val)
    if acc > max_acc:
        print('New MAX:', acc)
        max_acc = acc
        m.save("./model7.h5")

  print(max_acc)


