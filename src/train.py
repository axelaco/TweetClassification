import numpy as np
import keras
from preprocessing_git import data_preprocessing
from gensim.models import KeyedVectors
from keras.utils.np_utils import to_categorical
from word2vecUtils import afin, emojiValence, depechMood, emolex
import pickle

# EMBEDDING_DIM = 346
#
# corpora_train_3 = "../resources/data_train_3.csv"
# corpora_train_7 = "../resources/data_train_7.csv"


def getMaxLen(tweet3, tweet7):
    sequences_train_3 = keras.preprocessing.text.tokenizer.texts_to_sequences(tweet3)
    sequences_train_7 = keras.tokenizer.texts_to_sequences(tweet7)
    sequences = sequences_train_3 + sequences_train_7

    maxlen = 0
    for elt in sequences:
        if len(elt) > maxlen:
            maxlen = len(elt)

    return maxlen


def prepareData(corpora3, corpora7):
    tweet3, sentiment3 = data_preprocessing(corpora3, 'train')
    tweet7, sentiment7 = data_preprocessing(corpora7, 'test')

    all_tweet = tweet3.append(tweet7)

    tokenizer = keras.preprocessing.text.Tokenizer(filters=' ')
    tokenizer.fit_on_texts(all_tweet)
    word_index = tokenizer.word_index

    return word_index, tweet3, tweet7, sentiment3, sentiment7


def getTrainAndTestData7(tweet7, sentiment7, maxLen, tokenizer):
    sequences_train_7 = tokenizer.texts_to_sequences(tweet7)

    data_train_7 = keras.preprocessing.sequence.pad_sequences(sequences_train_7, maxlen=maxLen)
    indices_train_7 = np.arange(data_train_7.shape[0])
    data_train_7 = data_train_7[indices_train_7]

    labels_train_7 = to_categorical(np.asarray(sentiment7), 3)
    labels_train_7 = labels_train_7[indices_train_7]

    split_idx = int(len(data_train_7) * 0.70)
    x_train_7, x_val_7 = data_train_7[:split_idx], data_train_7[split_idx:]
    y_train_7, y_val_7 = labels_train_7[:split_idx], labels_train_7[split_idx:]

    return x_train_7, x_val_7,  y_train_7, y_val_7


def getTrainAndTestData3(tweet3, sentiment3, maxLen, tokenizer):
    sequences_train_3 = tokenizer.texts_to_sequences(tweet3)

    data_train_3 = keras.preprocessing.sequence.pad_sequences(sequences_train_3, maxlen=maxLen)
    indices_train_3 = np.arange(data_train_3.shape[0])
    data_train_3 = data_train_3[indices_train_3]

    labels_train_3 = to_categorical(np.asarray(sentiment3), 3)
    labels_train_3 = labels_train_3[indices_train_3]

    split_idx = int(len(data_train_3) * 0.70)
    x_train_3, x_val_3 = data_train_3[:split_idx], data_train_3[split_idx:]
    y_train_3, y_val_3 = labels_train_3[:split_idx], labels_train_3[split_idx:]

    return x_train_3, x_val_3, y_train_3, y_val_3

def concatenateEmbeding(word, word2vec, afinn, EV, depech, emolex):
    return word2vec.word_vec(word) + afin(afinn, word) + depechMood(depech, word) + emolex(emolex, word) + emojiValence(EV, word)


def createEmbedingMatrix(word_index, w2vpath, dim):
    word2vec = KeyedVectors.load_word2vec_format(w2vpath, binary=True)
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    oov = []
    oov.append((np.random.rand(dim) * 2.0) - 1.0)
    oov = oov / np.linalg.norm(oov)

    afinn = pickle.load('afinn')
    EV = pickle.load('emojiValence')
    depech = pickle.load('depech')
    emolex = pickle.load('emolex')

    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = concatenateEmbeding(word, word2vec, afinn, EV, depech, emolex)
        else:
            embedding_matrix[i] = oov

    return embedding_matrix

if __name__ == '__main__':
    word_index , t3, t7, s3, s7 = prepareData('../resources/data_train_3.csv', '../resources/data_train_7.csv')
    createEmbedingMatrix(word_index, '../resources/model2.bin', 346)

