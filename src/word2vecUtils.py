import pandas as pd
import numpy as np
import json
import pickle

def processAFIN():
    data = dict()
    f = open('../resources/AFINN.txt', 'r')
    for line in f.readlines():
        arr = line.strip().split('\t')
        word = arr[0]
        if word in data:
            continue
        else:
            data[word] = np.zeros(11)
            data[word][int(arr[1]) + 5] = 1
    return data

def afin(data, word):
    if word in data:
        return data[word]
    return np.zeros(11)


def processEmojiValence():
    data = dict()
    json_data = json.load(open('../resources/emoji_valence.json'))
    for json_elt in json_data:
        word = json_elt['emoji']
        data[word] = np.zeros(9)
        data[word][json_elt['polarity'] + 4] = 1
    return data

def emojiValence(data, word):
    if word in data:
        return data[word]
    return np.zeros(9)

def processDepechMode():
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

def depechMood(data, word):
    if word in data:
        return data[word]
    return np.zeros(12)

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

def processEmolex():
    indexes = {'anger': 0,'anticipation':1, 'disgust':2, 'fear':3,'joy':4,'negative':5,'positive':6,'sadness':7,'surprise':8,'trust':9}
    data = dict()
    f = open('../resources/Emolex.txt', 'r')
    for line in f.readlines():
        arr = line.strip().split('\t')
        word = arr[0]
        if (len(word) > 0):
            if word in data:
                data[word] = np.append(data[word], float(arr[2]))
            else:
                data[word] = np.array(float(arr[2]), dtype=np.float)
    f.close()
    # add emotion intensity
    f = open('../resources/Emolex_Intensity.txt', 'r')
    f.readline()
    for line in f.readlines():
        arr = line.strip().split('\t')
        word = arr[0]
        if (len(word) > 0):
            if word in data:
                data[word][indexes[arr[2]]] = arr[1]
            else:
                data[word] = np.zeros(9)
                data[word][indexes[arr[2]]] = arr[1]
    f.close()
    return data

def emolex(data, word):
    if word in data:
        return data[word]
    return np.zeros(9)

def processSentiment140():
    df = pd.read_csv('../resources/unigrams-pmilexicon.txt', sep='\t', names=['word', 'sentimentScore', 'numPositive', 'numNegative'])
    df = df.dropna()
    print(df.info())

def positiveExample(dataAfin, dataEmoji, dataDepechMood, dataEmolex):
    print(afin(dataAfin, 'abandon'))
    print(emojiValence(dataEmoji, '😠'))
    print(depechMood(dataDepechMood, 'absurdity'))
    print(processEmojiSentimentLexicon('😉'))
    print(emolex(dataEmolex, 'whimper'))

def negativeExample(dataAfin, dataEmoji, dataDepechMood, dataEmolex):
    print(afin(dataAfin, 'abandonsqa'))
    print(emojiValence(dataEmoji, 'iqf'))
    print(depechMood(dataDepechMood, 'absurditysqdqsd'))
    print(processEmojiSentimentLexicon('ia'))
    print(emolex(dataEmolex, 'whimperisaa'))


def saveDict(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def main():
    dataAfin = processAFIN()
    dataEmoji = processEmojiValence()
    dataDepechMood = processDepechMode()
    dataEmolex = processEmolex()

    saveDict(dataAfin, '../resources/embeding/afin')
    saveDict(dataDepechMood, '../resources/embeding/depech')
    saveDict(dataEmoji, '../resources/embeding/EV')
    saveDict(dataEmolex, '../resources/embeding/emolex')


    print("### Positive Example ###")
    positiveExample(dataAfin, dataEmoji, dataDepechMood, dataEmolex)
    print("\n### Negative Example ###")
    negativeExample(dataAfin, dataEmoji, dataDepechMood, dataEmolex)



# main()
