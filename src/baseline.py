import numpy as np
import pandas as pd

from options import *
from split_hashtag import split_hashtag_to_words


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
        for word in split_tweet:
            try:
                we_tweets[i] += words[word]
            except:
                if (not word.startswith("#")):
                    word = "#" + word
                tokens=split_hashtag_to_words(word)
                #print('before: ', word, ' after: ', split_hashtag_to_words(word))
                for token in tokens.split():
                    if((len(token) != 1) or (token == "a") or (token == "i")):
                        try:
                            we_tweets[i] += words[token]
                        except:
                            #print('Not found: ', token)
                            continue;
                continue;
        we_tweets[i] /= len(split_tweet)
    return we_tweets