import pandas as pd

from utils import read_file
from options import *

import fasttext

def fast_text(tweets, test_tweets):
    
    tweets['sentiment'] = change_label(tweets['sentiment'])
    
    write_tweets_with_fasttext_labels(tweets)

    classifier = fasttext.supervised(FASTTEXT_TRAIN_FILE, 'fasttext_model',\
                                                     label_prefix='__label__',\
                                                     epoch=WE_params['epochs'],\
                                                     dim=WE_params['we_features'],\
                                                     ws = WE_params['window_size'],\
                                                     lr = WE_params['learning_rate'])

    test_tweets = transform_test_tweets(test_tweets)

    labels = classifier.predict(test_tweets)
    labels = transform_labels(labels)
    return labels

def write_tweets_with_fasttext_labels(tweets):
    f = open(FASTTEXT_TRAIN_FILE,'w')
    for t,s in zip(tweets['tweet'], tweets['sentiment']):
        f.write((t.rstrip()+ ' '+s+'\n'))
    f.close()

def change_label(tweets):
    return tweets.apply(lambda row: '__label__'+str(row))

def transform_test_tweets(test_tweets):
    test_tweets_list = []
    for t in test_tweets['tweet']:
        test_tweets_list.append(t)
    return test_tweets_list

def transform_labels(labels):
    return [int(item) for sublist in labels for item in sublist]