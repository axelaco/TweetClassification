import pandas as pd
import numpy as np
import json
import pickle


def wv_afin(dict, word):
    if word in dict:
        return dict[word]
    return np.zeros(11)

def wv_emoji_valence(dict, word):
    if word in dict:
        return dict[word]
    return np.zeros(9)

def wv_depech_mood(dict, word):
    if word in dict:
        return dict[word]
    return np.zeros(12)

def wv_emoji_sentiment_lexicon(dict, word):
    if word in dict:
        return dict[word]
    return np.zeros(4)

def wv_opinion_lexicon_english(dict, word):
    if word in dict:
        return dict[word]
    return np.zeros(2)


def wv_emolex(dict, word):
    if word in dict:
        return dict[word]
    return np.zeros(10)
