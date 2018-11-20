import keras
import pickle
from  keras.optimizers import Adam
import sys
import os
from model import embedding_matrix_sentiment
import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Activation, Dense, Bidirectional,  Flatten, Input, GRU, GaussianNoise
from keras import regularizers
from sentiment_vectors import  wv_afin, wv_emoji_valence, wv_depech_mood, \
      wv_emoji_sentiment_lexicon, wv_opinion_lexicon_english, wv_emolex
from kutilities.layers import Attention


EMBEDDED_DIM = 348


def model(x_train, y_train, x_test, y_test, embedding_layer):
  model = Sequential()

  model.add(embedding_layer)
  model.add(GaussianNoise(0.3))
  model.add(Dropout(0.3))
  model.add(Bidirectional(LSTM(150, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0), return_sequences=True)))
  model.add(Dropout(0.3))
  model.add(Bidirectional(LSTM(150, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0), return_sequences=True)))
  model.add(Dropout(0.3))
  model.add(Attention())
  model.add(Dense(2, activity_regularizer=regularizers.l2(0.0001)))
  model.add(Activation('softmax'))
  model.compile(optimizer=Adam(clipnorm=1, lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
  model.summary()

  model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=12, batch_size=50)

  model.save("./model_two_classes.h5")



def train_two_class():
  # path = "/Users/franckthang/Work/PersonalWork/sentiment-analysis/floyd-dataset"
  path = "/floyd/input/dataset"
  dataset = os.path.join(path, "text_preprocessed.csv")

  # resources  = "/Users/franckthang/Work/PersonalWork/sentiment-analysis/resources"
  word2vec_path = os.path.join(path, 'datastories.twitter.300d.pickle')

  # sentiment, text
  # sentiment 0 = negative
  # sentiment 4 = positive
  df = pd.read_csv(dataset).dropna()
  y_dataset, x_dataset = df['sentiment'].values, df['text'].values
  y_dataset = [x % 3 for x in y_dataset]

  # Creating tokenizer, so we can access to word index
  tokenizer = keras.preprocessing.text.Tokenizer(filters=' ')
  tokenizer.fit_on_texts(x_dataset)
  word_index = tokenizer.word_index

  sequences = tokenizer.texts_to_sequences(x_dataset)
  MAX_SEQUENCE_LENGTH = len(max(sequences, key=len))
  x_dataset = keras.preprocessing.sequence.pad_sequences(sequences, \
    maxlen=MAX_SEQUENCE_LENGTH)
  y_dataset = to_categorical(np.asarray(y_dataset),2)

  # Loading word2vec 330m tweets
  embedding_matrix = embedding_matrix_sentiment(word_index, word2vec_path)
  embedding_layer = Embedding(len(word_index) + 1,
                          EMBEDDED_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          trainable=False, name='embedding_layer')
  X_train, X_test, y_train, y_test = train_test_split(
    x_dataset, y_dataset, random_state=42)


  model(X_train, y_train, X_test, y_test, embedding_layer)




train_two_class()
