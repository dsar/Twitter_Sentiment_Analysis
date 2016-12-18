import numpy as np
import pandas as pd
from build_embeddings import get_embeddings_dictionary

from options import *
from split_hashtag import split_hashtag_to_words

def we_mean(tweets, test_tweets):
    """
    DESCRIPTION: 
            Given the calculated word embedings (of some dimension d) of the training and test set, 
            this function returns the tweet embeddings of the same d dimension by just averaging the
            vectors of each word in the same tweet. This is done for tweets and test_tweets pandas Dataframes.
    INPUT: 
            tweets: Dataframe of a set of tweets as a python strings
            test_tweets: Dataframe of a set of test tweets as a python strings
    OUTPUT: 
            we_tweets: tweet embeddings of d dimension
            we_test_tweets: test tweet embeddings of d dimension
    """
    words = get_embeddings_dictionary(tweets)
    print('\nBuilding tweets Embeddings')
    we_tweets = average_vectors(tweets, words)
    print('Building test tweets Embeddings')
    we_test_tweets = average_vectors(test_tweets, words)
    return we_tweets, we_test_tweets

def average_vectors(tweets, words):
    """
    DESCRIPTION: 
            Given a pandas Dataframe of tweets and the trained word embedings (of some dimension d)
            this function returns the tweet embeddings of the same d dimension by just averaging the
            vectors of each word in the same tweet. 
    INPUT: 
            tweets: Dataframe of a set of tweets as a python strings
            words: python dictionary (word, vector of embeddings of the corresponding word)
    OUTPUT: 
            we_tweets: tweet embeddings of d dimension
    """
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
