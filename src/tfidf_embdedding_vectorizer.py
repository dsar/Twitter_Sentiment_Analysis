import collections
import numpy as np
import pandas as pd
from vectorizer import init_tfidf_vectorizer
from utils import get_embeddings_dictionary

from options import *
from split_hashtag import split_hashtag_to_words


def tfidf_embdedding_vectorizer(tweets, test_tweets):
    words = get_embeddings_dictionary()
    print('building train tfidf')
    vectorizer_params['tokenizer'] = None
    tfidf = init_tfidf_vectorizer()
    X = tfidf.fit_transform(tweets['tweet'].append(test_tweets['tweet']))
    
    print('train tweets: building (TF-IDF-weighted) WEs')
    we_tweets = average_vectors(tweets, words, tfidf, X[:tweets.shape[0]])
    print('test tweets: building (TF-IDF-weighted) WEs')
    we_test_tweets = average_vectors(test_tweets, words, tfidf, X[tweets.shape[0]:])
    return we_tweets, we_test_tweets
    
def average_vectors(tweets, words, tfidf, X):
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
