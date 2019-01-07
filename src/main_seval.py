import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Bidirectional, Dropout, LeakyReLU
import numpy as np
import keras
from preprocessing_git import data_preprocessing, standardization
from gensim.models import KeyedVectors
from keras.utils.np_utils import to_categorical
import pickle
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
from keras import optimizers
from keras.models import load_model
import json, argparse, os
import re
import pandas as pd
import io
import sys
from keras.models import Model
from train import get_max_len, createEmbedingMatrix, createEmbeddingMatrixGlove
from preprocessing_git import data_preprocessing_teacher


NUM_CLASSES = 4                 # Number of classes - Happy, Sad, Angry, Others
EMBEDDING_DIM = 200               # The dimension of the word embeddings
BATCH_SIZE = 128                  # The batch size to be chosen for training the model.
LSTM_DIM = 100                    # The dimension of the representations learnt by the LSTM model
DROPOUT = 0.3                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = 12                  # Number of epochs to train a model for
LEARNING_RATE = 0.001


from preprocessing_git import data_preprocessing_semeval
def buildModel(embeddingMatrix, MAX_SEQUENCE_LENGTH):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    x1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input1')
    x2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input2')
    x3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input3')

    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    emb1 = embeddingLayer(x1)
    emb2 = embeddingLayer(x2)
    emb3 = embeddingLayer(x3)

    lstm = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT))

    lstm1 = lstm(emb1)
    lstm2 = lstm(emb2)
    lstm3 = lstm(emb3)

    inp = Concatenate(axis=-1)([lstm1, lstm2, lstm3])

    inp = Reshape((3, 2*LSTM_DIM, )) (inp)

    lstm_up = LSTM(LSTM_DIM, dropout=DROPOUT)

    out = lstm_up(inp)

    out = Dense(NUM_CLASSES, activation='softmax')(out)
    
    adam = optimizers.adam(lr=LEARNING_RATE)
    model = Model([x1,x2,x3],out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    print(model.summary())
    return model

def lstm_model(x_train_3, y_train_3, x_val_3, y_val_3, embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(100))
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(Dense(4, activity_regularizer=l2(0.0001)))
    model.add(Activation('softmax'))
    model.compile(optimizer=Adam(clipnorm=1, lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.6, patience=5, \
                          verbose=1, mode='auto')

    filepath="lstm_model-{epoch:02d}-{val_acc:.2f}.hdf5"
    modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    callbacks_list = [modelCheckpoint]
    model.fit(x_train_3, y_train_3, validation_data=(x_val_3, y_val_3),epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    model.save("./lstm_model.h5")

def lstm_model_with_weight(embedding_layer, weight_filename):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(100))
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(Dense(4, activity_regularizer=l2(0.0001)))
    model.add(Activation('softmax'))
    model.load_weights(weight_filename)
    model.compile(optimizer=Adam(clipnorm=1, lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    return model

def b_lstm_model(x_train_3, y_train_3, x_val_3, y_val_3, embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(GaussianNoise(0.3))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150, recurrent_dropout=0.3, kernel_regularizer=l2(0), return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150, recurrent_dropout=0.3, kernel_regularizer=l2(0), return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dense(2, activity_regularizer=l2(0.0001)))
    model.add(Activation('softmax'))
    model.compile(optimizer=Adam(clipnorm=1, lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.6, patience=5, \
                          verbose=1, mode='auto')

    filepath="b_lstm-model-{epoch:02d}-{val_acc:.2f}.hdf5"
    modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    callbacks_list = [modelCheckpoint]
    model.fit(x_train_3, y_train_3, validation_data=(x_val_3, y_val_3),epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    model.save("./b_lstm-model.h5")

def b_lstm_model_with_weight(embedding_layer, weight_filename):
    model = Sequential()
    model.add(embedding_layer)
    model.add(GaussianNoise(0.3))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150, recurrent_dropout=0.3, kernel_regularizer=l2(0), return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(150, recurrent_dropout=0.3, kernel_regularizer=l2(0), return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Attention())
    model.add(Dense(2, activity_regularizer=l2(0.0001)))
    model.add(Activation('softmax'))
    model.load_weights(weight_filename)
    model.compile(optimizer=Adam(clipnorm=1, lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.summary()
    return model

def bc_lstm_model_train():
    trainIndices, u1_train, u2_train, u3_train, labels  = data_preprocessing_semeval('../resources/train.txt', 'train')
    trainIndices, u1_test, u2_test, u3_test  = data_preprocessing_semeval('../resources/train.txt', 'test')
    print("Extracting tokens...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(u1_train + u2_train + u3_train)
    u1_trainSequences, u2_trainSequences, u3_trainSequences = tokenizer.texts_to_sequences(u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train)
    u1_testSequences, u2_testSequences, u3_testSequences = tokenizer.texts_to_sequences(u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test)

    MAX_SEQUENCE_LENGTH = get_max_len([u1_train, u2_train, u3_train], tokenizer)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embedding_matrix = createEmbedingMatrix(wordIndex, '../resources/model.kv', EMBEDDING_DIM)

    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))

    np.random.shuffle(trainIndices)

    u1_data = u1_data[trainIndices]
    u2_data = u2_data[trainIndices]
    u3_data = u3_data[trainIndices]



    labels = labels[trainIndices]


        # Perform k-fold cross validation
    metrics = {"accuracy" : [],
                "microPrecision" : [],
                "microRecall" : [],
                "microF1" : []}

    model = buildModel(embedding_matrix, MAX_SEQUENCE_LENGTH)

    model.fit([u1_data, u2_data, u3_data], labels,  epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    model.save('semeval_phd.h5')

def train_model(network_model):
    x_train, y_train = data_preprocessing_teacher('../resources/train.txt', 'True')
    x_test = data_preprocessing_teacher('../resources/test.txt', 'False')
    all_tweet = x_train.append(x_test)
    tokenizer = Tokenizer(filters=' ')
    tokenizer.fit_on_texts(all_tweet)
    word_index = tokenizer.word_index
    sequences_train = tokenizer.texts_to_sequences(x_train)
    sequences_test = tokenizer.texts_to_sequences(x_test)
    MAX_SEQUENCE_LENGTH = get_max_len([x_test, x_train], tokenizer)
    data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
    indices_train = np.arange(data_train.shape[0])
    data_train = data_train[indices_train]
    indices_test = np.arange(data_test.shape[0])
    data_test = data_test[indices_test]
    nb_words=len(word_index)+1

    y_train = to_categorical(np.asarray(y_train), 4)
    embedding_matrix =  createEmbeddingMatrixGlove(word_index, '../resources/model.kv', EMBEDDING_DIM)

    embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True, name='embedding_layer')

    split_idx = int(len(x_train)*0.95)
    x_train, x_val = data_train[:split_idx], data_train[split_idx:]
    y_train, y_val = y_train [:split_idx], y_train[split_idx:]
    network_model(x_train, y_train, x_val, y_val, embedding_layer)

def validation_model(modelPath):
    x_train, y_train = data_preprocessing_teacher('../resources/train.txt', 'True')
    x_test= data_preprocessing_teacher('../resources/test.txt', 'False')

    all_tweet = x_train.append(x_test)

    tokenizer = Tokenizer(filters=' ')
    tokenizer.fit_on_texts(all_tweet)
    word_index = tokenizer.word_index

    nb_words=len(word_index)+1
    sequences_train = tokenizer.texts_to_sequences(x_train)
    sequences_test = tokenizer.texts_to_sequences(x_test)

    MAX_SEQUENCE_LENGTH = get_max_len([x_train, x_test], tokenizer)
    sequences = sequences_train + sequences_test
    data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    indices_train = np.arange(data_train.shape[0])
    data_train = data_train[indices_train]

    indices_test = np.arange(data_test.shape[0])
    data_test = data_test[indices_test]

    embedding_matrix = createEmbeddingMatrixGlove(word_index, '../resources/model.kv', EMBEDDING_DIM)

    embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True, name='embedding_layer')
 
    f = open("./test.txt", "w")
    f.write("id\tturn1\tturn2\tturn3\tlabel\n")
    nn_model = create_custom_model(embedding_layer, modelPath)
    r = nn_model.predict(data_test)
    data = pd.read_csv("../resources/test.txt", sep='\t', encoding='utf-8', names=['id','turn1','turn2','turn3'])
    for d in range(1,len(data)):
        i=d-1 
        idx=np.argmax(r[i])
        if (idx==3):
            label="others"
        elif(idx==2):
            label="sad"
        elif(idx==1):
            label="happy"
        elif(idx==0):
            label="angry"
        f.write(str(data["id"][d])+"\t"+str(data["turn1"][d])+"\t"+str(data["turn2"][d])+"\t"+str(data["turn3"][d])+"\t"+label+"\n")
    f.close()

def validation_bc_lstm_model(modelFile):
    print("Processing training data...")
    train_indices, u1_train, u2_train, u3_train, labels = data_preprocessing_semeval('../resources/train.txt', 'train')
    print("Processing test data...")
    test_indices, u1_test, u2_test, u3_test = data_preprocessing_semeval('../resources/test.txt', 'test')

    print("Extracting tokens...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(u1_train + u2_train + u3_train)
    u1_trainSequences, u2_trainSequences, u3_trainSequences = tokenizer.texts_to_sequences(u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train)
    u1_testSequences, u2_testSequences, u3_testSequences = tokenizer.texts_to_sequences(u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test)
    MAX_SEQUENCE_LENGTH = get_max_len([u1_train, u2_train, u3_train], tokenizer)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embedding_matrix = createEmbedingMatrix(wordIndex, '../resources/model.kv', EMBEDDING_DIM)

    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)

    u1_testData, u2_testData, u3_testData = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    model = load_model(modelFile)
    model.summary()
    predictions = model.predict([u1_testData, u2_testData, u3_testData], batch_size=50)
    predictions = predictions.argmax(axis=1)
    with io.open('solution_teacher.csv', "w", encoding="utf8") as fout:
            fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
            with io.open('../resources/test.txt', encoding="utf8") as fin:
                fin.readline()
                for lineNum, line in enumerate(fin):
                    fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                    fout.write(label2emotion[predictions[lineNum]] + '\n')
            print('Completed. Model parameters: ')


def main():
    train_model(lstm_model)