import numpy as np
import pandas as pd

from options import *

def baseline(tweets, test_tweets):
    we = np.load(DATA_PATH+'embeddings.npy')
    print('we shape', we.shape)

    vocab_file = open(DATA_PATH+'vocab_cut.txt', "r")
    words = {} #key= word, value=embeddings
    for i, line in enumerate(vocab_file):
        words[line.rstrip()] = we[i]
	
    print('building tweets WE')
    we_tweets = average_vectors(tweets, words)

    print('building test tweets WE')
    we_test_tweets = average_vectors(test_tweets, words)

    return we_tweets, we_test_tweets


def average_vectors(tweets, words):
    we_tweets = np.zeros((tweets.shape[0],WE_params['we_features']))
    i = 0
    for tweet in tweets['tweet']:
        split_tweet = tweet.split()
        vec = np.zeros(WE_params['we_features'])
        j = 0.0
        for word in split_tweet:
            try:
                vec += words[word]
                j += 1.0
            except:
                continue;
        if(j != 0.0):
            we_tweets[i] = vec/j
        i+=1
    return we_tweets