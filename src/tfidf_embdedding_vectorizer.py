import collections
import numpy as np
import pandas as pd
from vectorizer import init_tfidf_vectorizer
from build_embeddings import get_embeddings_dictionary

from options import *
from split_hashtag import split_hashtag_to_words


def tfidf_embdedding_vectorizer(tweets, test_tweets):
    """
    DESCRIPTION: 
            Given the calculated word embedings (of some dimension d) of the training and test set, 
            this function returns the tweet embeddings of the same d dimension by just averaging the
            vectors of each word in the same tweet and multipliying the corresponding word embedding
            with it's tfidf value.
            This is done for tweets and test_tweets pandas Dataframes.
    INPUT: 
            tweets: Dataframe of a set of tweets as a python strings
            test_tweets: Dataframe of a set of test tweets as a python strings
    OUTPUT: 
            we_tweets: tweet embeddings of d dimension
            we_test_tweets: test tweet embeddings of d dimension
    """
    words = get_embeddings_dictionary(tweets)
    print('building train tfidf')
    algorithm['options']['TFIDF']['tokenizer'] = None
    tfidf = init_tfidf_vectorizer()
    X = tfidf.fit_transform(tweets['tweet'])
    print('train tweets: building (TF-IDF-weighted) WEs')
    we_tweets = average_vectors(tweets, words, tfidf, X)
    print('test tweets: building (TF-IDF-weighted) WEs')
    tfidf = init_tfidf_vectorizer()
    X = tfidf.fit_transform(test_tweets['tweet'])
    we_test_tweets = average_vectors(test_tweets, words, tfidf, X)
    return we_tweets, we_test_tweets
    
def average_vectors(tweets, words, tfidf, X):
    """
    DESCRIPTION: 
            Given a pandas Dataframe of tweets and the trained word embedings (of some dimension d)
            this function returns the tweet embeddings of the same d dimension by just averaging the
            vectors of each word in the same tweet and multipliying the corresponding word embedding
            with it's tfidf value. 
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
                try:
                    weight = X[i,tfidf.vocabulary_[word]]
                except:                    
                    weight = 1
                
                we_tweets[i] += words[word] * weight
                foundEmbeddings+=1
            except:
                if (not word.startswith("#")):
                    word = "#" + word
                tokens=split_hashtag_to_words(word)
                #print('before: ', word, ' after: ', split_hashtag_to_words(word))
                for token in tokens.split():
                    if((len(token) != 1) or (token == "a") or (token == "i")):
                        try:
                            try:
                                weight = X[i,tfidf.vocabulary_[token]]
                            except:                    
                                weight = 1
                            we_tweets[i] += words[token] * weight
                            foundEmbeddings+=1
                        except:
                            #print('Not found: ', token)
                            continue;
                continue;
        if (foundEmbeddings != 0):
            we_tweets[i] /= foundEmbeddings
    return we_tweets
