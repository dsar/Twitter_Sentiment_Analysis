import numpy as np
import pandas as pd
from utils import get_embeddings_dictionary

from options import *
from split_hashtag import split_hashtag_to_words

def baseline(tweets, test_tweets):
    words = get_embeddings_dictionary(tweets)
    print('building tweets WE')
    we_tweets = average_vectors(tweets, words)
    print('building test tweets WE')
    we_test_tweets = average_vectors(test_tweets, words)
    return we_tweets, we_test_tweets

def average_vectors(tweets, words):
    we_tweets = np.zeros((tweets.shape[0], len(next(iter(words.values())))))
    for i, tweet in enumerate(tweets['tweet']):
        try:
            split_tweet = tweet.split()
        except:
            continue;

        foundEmbeddings = 0
        for word in split_tweet:
            try:
                we_tweets[i] += words[word]
                foundEmbeddings+=1
            except:
                if (not word.startswith("#")):
                    word = "#" + word
                tokens=split_hashtag_to_words(word)
                #print('before: ', word, ' after: ', split_hashtag_to_words(word))
                for token in tokens.split():
                    if((len(token) != 1) or (token == "a") or (token == "i")):
                        try:
                            we_tweets[i] += words[token]
                            foundEmbeddings+=1
                        except:
                            #print('Not found: ', token)
                            continue;
                continue;
        if (foundEmbeddings != 0):
            we_tweets[i] /= foundEmbeddings
    return we_tweets
