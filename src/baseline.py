import numpy as np
import pandas as pd

from options import *

def baseline(tweets, test_tweets):
    words = get_embeddings_dictionary()
    print('building tweets WE')
    we_tweets = average_vectors(tweets, words)
    print('building test tweets WE')
    we_test_tweets = average_vectors(test_tweets, words)
    return we_tweets, we_test_tweets

def get_embeddings_dictionary():
    words = {} #key= word, value=embeddings
    trainEmbeddings = False
    if (trainEmbeddings):
        we = np.load(DATA_PATH+EMBEDDINGS_FILE)
        print('we shape', we.shape)
        vocab_file = open(DATA_PATH+'vocab_cut.txt', "r")
        for i, line in enumerate(vocab_file):
            words[line.rstrip()] = we[i]
    else:
        with open(GLOVE_DATA_PATH+EMBEDDINGS_FILE_200, "r") as f:
            for line in f:
                tokens = line.strip().split()
                words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    return words

def average_vectors(tweets, words):
    we_tweets = np.zeros((tweets.shape[0], len(next(iter(words.values())))))
    for i, tweet in enumerate(tweets['tweet']):
        split_tweet = tweet.split()
        #j = 0.0
        for word in split_tweet:
            try:
                we_tweets[i] += words[word]
                #j += 1.0
            except:
                continue;
        #if(j != 0.0):
        #    we_tweets[i] /= j
    return we_tweets