import pandas as pd

from utils import read_file
from options import *

import fasttext

def fast_text(tweets, test_tweets):
    
    tweets['sentiment'] = change_label(tweets['sentiment'])
    
    write_tweets_with_fasttext_labels(tweets)

    classifier = fasttext.supervised(FASTTEXT_TRAIN_FILE, FASTTEXT_MODEL,\
                                                     label_prefix='__label__',\
                                                     epoch=algorithm['params']['epochs'],\
                                                     dim=algorithm['params']['we_features'],\
                                                     ws = algorithm['params']['window_size'],\
                                                     lr = algorithm['params']['learning_rate'])

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