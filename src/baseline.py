import numpy as np
import pandas as pd

from options import *

def baseline(tweets, test_tweets):
    we = np.load(DATA_PATH+'embeddings.npy')
    print('we shape', we.shape)

    vocab_file = open(DATA_PATH+'vocab_cut.txt', "r")
    line_index = 0
    words = {} #key= word, value=index
    for line in vocab_file:
        words[line.rstrip()] = line_index
        line_index += 1

    print('vocab size:', line_index)
	
    print('building tweets WE')
    we_tweets = np.zeros((tweets.shape[0],WE_params['we_features']))
    i = 0
    for tweet in tweets['tweet']:
        split_tweet = tweet.split()
        vec = np.zeros(WE_params['we_features'])
        for word in split_tweet:
            try:
                vec += we[words[word]]
            except:
                vec += 0
        we_tweets[i] = vec
        i+=1


    print('building test tweets WE')
    we_test_tweets = np.zeros((test_tweets.shape[0],WE_params['we_features']))
    i = 0
    for tweet in test_tweets['tweet']:
        split_tweet = tweet.split()
        vec = np.zeros(WE_params['we_features'])
        for word in split_tweet:
            try:
                vec += we[words[word]]
            except:
                vec += 0
        we_test_tweets[i] = vec
        i+=1

    return we_tweets, we_test_tweets

