import numpy as np
import keras
from preprocessing_git import data_preprocessing
from gensim.models import KeyedVectors
from keras.utils.np_utils import to_categorical
import pickle
import sys
from  gensim.models import Word2Vec
import word2vecUtils
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Bidirectional,  Flatten, Input, GRU, GaussianNoise
from keras import regularizers
import matplotlib as mpl
from keras.optimizers import Adam
# DEBUG purpose
#import importlib
#importlib.reload(word2vecUtils)
from word2vecUtils import afin, emojiValence, depechMood, emolex, \
  emojiSentimentLexicon, opinionLexiconEnglish


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

def get_train_test(tweet, sentiment, max_len, tokenizer):
    sequences_train = tokenizer.texts_to_sequences(tweet)

    data_train = keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_len)
    indices_train = np.arange(data_train.shape[0])
    data_train = data_train[indices_train]

    labels_train = to_categorical(np.asarray(sentiment), 3)
    labels_train = labels_train[indices_train]

    split_idx = int(len(data_train) * 0.70)
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

def model1(x_train_3, y_train_3,x_val_3, y_val_3, embedding_layer):
  model1 = Sequential()
  model1.add(embedding_layer)
  model1.add(GaussianNoise(0.2))
  model1.add(Dropout(0.3))
  model1.add(Bidirectional(LSTM(150, recurrent_dropout=0.25, return_sequences=True)))
  model1.add(Dropout(0.5))
  model1.add(Bidirectional(LSTM(150, recurrent_dropout=0.25)))
  model1.add(Dropout(0.5))
  model1.add(Dense(3, activation='softmax',kernel_regularizer=regularizers.l2(0.0001),
                activity_regularizer=regularizers.l2(0.0001)))
  model1.compile(loss='categorical_crossentropy',
			      optimizer=Adam(lr=0.01),
			      metrics=['acc'])
  model1.summary()
  history=model1.fit(x_train_3, y_train_3, validation_data=(x_val_3, y_val_3),epochs=6, batch_size=50)
  model1.save("./model1.h5")

def model2_bis(x_train_3, y_train_3,x_val_3, y_val_3, embedding_layer):
    model2 = Sequential()
    model2.add(embedding_layer)
    model2.add(Bidirectional(LSTM(32, recurrent_dropout=0.25, return_sequences=True)))
    model2.add(Dropout(0.5))
    model2.add(Bidirectional(LSTM(32, recurrent_dropout=0.25,  return_sequences=True)))
    model2.add(Dropout(0.5))
    model2.add(Flatten())
    model2.add(Dense(3, activation='softmax'))
    model2.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['acc'])
    model2.summary()
    history=model2.fit(x_train_3, y_train_3, validation_data=(x_val_3, y_val_3),epochs=6, batch_size=50)
    model2.save("./model2.h5")

def model2(x_train_3, y_train_3,x_val_3, y_val_3, embedding_layer):

	model2 = Sequential()
	model2.add(embedding_layer)
	model2.add(LSTM(32))
	model2.add(Dropout(0.2))
	model2.add(Dense(32, activation='relu'))
	model2.add(Dropout(0.2))
	model2.add(Dense(3, activation='softmax'))
	model2.compile(loss='categorical_crossentropy',
			      optimizer='Adam',
			      metrics=['acc'])
	model2.summary()
	history=model2.fit(x_train_3, y_train_3, validation_data=(x_val_3, y_val_3),epochs=6, batch_size=50)
	model2.save("./model2.h5")

if __name__ == '__main__':
  corpora_train_3 = '../resources/data_train_3.csv'
  corpora_train_7 = '../resources/data_train_7.csv'
  corpora_test_7 = "'../resources/data_test_7.csv'"


  word_index, tokenizer, tweet3, tweet7, sentiment3, sentiment7 = prepareData(corpora_train_3, corpora_train_7)
  #model = Word2Vec.load('../resources/model_5M.bin')
  #saveKeyedVectors('../resources/model2.kv', model)
  
  MAX_SEQUENCE_LENGTH = get_max_len([tweet3, tweet7], tokenizer)

  embedding_matrix = createEmbedingMatrix(word_index, '../resources/model2.kv', EMBEDDING_DIM)

  x_train_3, x_val_3, y_train_3, y_val_3 = get_train_test(tweet3, sentiment3, MAX_SEQUENCE_LENGTH, tokenizer)
  embedding_layer = Embedding(len(word_index) + 1,
                          EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=False, name='embedding_layer')

  model2_bis(x_train_3, y_train_3,x_val_3, y_val_3, embedding_layer)