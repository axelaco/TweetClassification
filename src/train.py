import numpy as np
import keras
from preprocessing_git import data_preprocessing, standardization
from gensim.models import KeyedVectors
from keras.utils.np_utils import to_categorical
import pickle
import pandas as pd
from scipy.stats import pearsonr
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import LSTM, Dropout, Dense, Activation, Bidirectional,  Flatten, Input, GRU, GaussianNoise
from keras.models import load_model
from kutilities.layers import Attention
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
import re
import io
import os
# DEBUG purpose
from word2vecUtils import afin, emojiValence, depechMood, emolex, \
  emojiSentimentLexicon, opinionLexiconEnglish
from sklearn.model_selection import StratifiedKFold


#word2vec_tweet = pickle.load(open('../resources/datastories.twitter.300d.pickle', 'rb'))
EMBEDDING_DIM = 300

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


def get_dataset(tweet, sentiment, max_len, tokenizer, size):
    sequences_train = tokenizer.texts_to_sequences(tweet)
    data_train = pad_sequences(sequences_train, maxlen=max_len)
    indices_train = np.arange(data_train.shape[0])
    data_train = data_train[indices_train]
    labels_train = to_categorical(np.asarray(sentiment), size)
    labels_train = labels_train[indices_train]

    return data_train, labels_train


def get_train_test(tweet, sentiment, max_len, tokenizer, size):
    sequences_train = tokenizer.texts_to_sequences(tweet)

    data_train = pad_sequences(sequences_train, maxlen=max_len)
    indices_train = np.arange(data_train.shape[0])
    data_train = data_train[indices_train]

    labels_train = to_categorical(np.asarray(sentiment), size)
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

def createEmbeddingMatrixGlove(word_index, w2vpath, dim):
    oov = []
    oov.append((np.random.rand(dim) * 2.0) - 1.0)
    oov = oov / np.linalg.norm(oov)
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join('../resources', 'glove.840B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
           # print(values)
            word = values[0]
            embeddingVector = np.array([float(val) for val in values[1:]])
            embeddingsIndex[word] = embeddingVector
        print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(wordIndex) + 1, dim))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            embeddingMatrix[i] = oov
    return embeddingMatrix

def createEmbedingMatrix(word_index, w2vpath, dim):
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    oov = []
    oov.append((np.random.rand(dim) * 2.0) - 1.0)
    oov = oov / np.linalg.norm(oov)

    path = "../resources/embeding"
    EMBEDDING_FILE="~/Téléchargements/GoogleNews-vectors-negative300.bin"
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

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
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)#np.concatenate(concat)
        else:
            embedding_matrix[i] = oov

    return embedding_matrix


def model(x_train_3, y_train_3, x_val_3, y_val_3, embedding_layer):
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
    model2.fit(x_train_3, y_train_3, validation_data=(x_val_3, y_val_3),epochs=12, batch_size=50)
    model2.save("./model2.h5")

def model_final(modelPath, x_train_7, y_train_7, x_val_7, y_val_7):
    model = load_model(modelPath, custom_objects={"Attention": Attention})
    model.summary()
    model.layers.pop()
    model.layers.pop()
    model.add(Dense(150,activation='relu',name='dense1'))
    model.add(Dense(7,activation='softmax',name='dense2'))
    model.summary()
    model.compile(optimizer=Adam(clipnorm=1, lr=0.001), loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_train_7, y_train_7,   validation_data=(x_val_7,y_val_7), epochs=18, batch_size=64)
    return model


def pearson_score(model, x_val_7, y_val_7, str):
    pred = model.predict(x_val_7)
    y_pred = np.argmax(pred, axis=1)
    score = pearsonr(np.argmax(y_val_7, axis=1), y_pred)[0]
    print('score for', str, score)
    return score


def process_test(path):
    df = pd.read_csv(path, sep="\t")
    s = df['Intensity Class'].values
    data = []
    for i in s:
        data.append(int(re.search('[-0-9]+(?=:)', i).group(0)))

    tweet = df['Tweet'].apply(lambda x: standardization(x))
    return tweet, data


def fill_csv(csvpath, modelpath, tokenizer, max_len):
    scores = {-3: ': very negative emotional state can be inferred',
              -2: ': moderately negative emotional state can be inferred',
              -1: ': slightly negative emotional state can be inferred',
              0: ': neutral or mixed emotional state can be inferred',
              1: ': slightly positive emotional state can be inferred',
              2: ': moderately positive emotional state can be inferred',
              3: ': very positive emotional state can be inferred'}

    df = pd.read_csv(csvpath, sep="\t")

    tweet = df['Tweet'].apply(lambda x: standardization(x))
    sequences_train = tokenizer.texts_to_sequences(tweet)
    data_train = pad_sequences(sequences_train, maxlen=max_len)

    model = load_model(modelpath, custom_objects={"Attention": Attention})

    pred = model.predict(data_train)
    preds = np.argmax(pred, axis=1)
    preds = [x - 3 for x in preds]
    preds = [str(x) + scores[x] for x in preds]
    df['Intensity Class'] = preds

    df.to_csv('../resources/prediction2.txt', index=None, sep='\t', mode='a')


def train_model():
    corpora_train_3 = '../resources/data_train_3.csv'
    corpora_train_7 = '../resources/data_train_7.csv'

    word_index, tokenizer, tweet3, tweet7, sentiment3, sentiment7 = prepareData(corpora_train_3, corpora_train_7)

    sentiment7 = [x + 3 for x in sentiment7]

    MAX_SEQUENCE_LENGTH = get_max_len([tweet3, tweet7], tokenizer)

    embedding_matrix = createEmbedingMatrix(word_index, '../resources/model2.kv', EMBEDDING_DIM)

    x_dataset_7, y_dataset_7 = get_dataset(tweet7, sentiment7, MAX_SEQUENCE_LENGTH, tokenizer, 7)
    x_train_3, x_val_3, y_train_3, y_val_3 = get_train_test(tweet3, sentiment3, MAX_SEQUENCE_LENGTH, tokenizer, 3)

    x_data_dev, y_data_dev = process_test('../resources/2018-Valence-oc-En-dev.txt')

    y_data_dev = [x + 3 for x in y_data_dev]

    x_val_dev, y_val_dev = get_dataset(x_data_dev, y_data_dev, MAX_SEQUENCE_LENGTH, tokenizer, 7)

    embedding_layer = Embedding(len(word_index) + 1,
                          EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH,
                          mask_zero=True,
                          trainable=False, name='embedding_layer')

    model(x_train_3, y_train_3, x_val_3, y_val_3, embedding_layer)

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    max_acc = 0

    for train_index, test_index in skf.split(x_dataset_7, np.argmax(y_dataset_7, axis=-1)):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x_dataset_7[train_index], x_dataset_7[test_index]
        y_train, y_test = y_dataset_7[train_index], y_dataset_7[test_index]
        m = model_final("./model2.h5", x_train, y_train, x_test, y_test)
        acc = pearson_score(m, x_val_dev, y_val_dev, 'dev data')
        pearson_score(m, x_dataset_7, y_dataset_7, 'train data')
        if acc > max_acc:
            print('New MAX:', acc)
            max_acc = acc
            m.save("./model7.h5")

    return tokenizer, MAX_SEQUENCE_LENGTH


