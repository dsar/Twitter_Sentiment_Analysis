import collections
import numpy as np
import pandas as pd
from vectorizer import init_tfidf_vectorizer

from options import *
from split_hashtag import split_hashtag_to_words


def tfidf_embdedding_vectorizer(tweets, test_tweets):
    words = get_embeddings_dictionary()
    print('building train tfidf')
    vectorizer_params['tokenizer'] = None
    tfidf = init_tfidf_vectorizer()
    tfidf.fit_transform(tweets['tweet'])
    max_idf = max(tfidf.idf_)
    word2weight = collections.defaultdict(lambda: max_idf,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    print('building tweets WE')
    we_tweets = average_vectors(tweets, words, word2weight)
    print('building test tweets WE')
    tfidf = init_tfidf_vectorizer()
    tfidf.fit_transform(test_tweets['tweet'])
    max_idf = max(tfidf.idf_)
    word2weight = collections.defaultdict(lambda: max_idf,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    we_test_tweets = average_vectors(test_tweets, words, word2weight)
    return we_tweets, we_test_tweets

def get_embeddings_dictionary():
    words = {} #key= word, value=embeddings
    trainEmbeddings = False
    if (trainEmbeddings):
        we = np.load(EMBEDDINGS_FILE)
        print('we shape', we.shape)
        vocab_file = open(DATA_PATH+'vocab_cut.txt', "r")
        for i, line in enumerate(vocab_file):
            words[line.rstrip()] = we[i]
    else:
        with open(EMBEDDINGS_FILE_200, "r") as f:
            for line in f:
                tokens = line.strip().split()
                words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    return words

def average_vectors(tweets, words, word2weight):
    we_tweets = np.zeros((tweets.shape[0], len(next(iter(words.values())))))
    for i, tweet in enumerate(tweets['tweet']):
        try:
            split_tweet = tweet.split()
        except:
            continue;

        foundEmbeddings = 0
        for word in split_tweet:
            try:
                we_tweets[i] += words[word] * word2weight[word]
                foundEmbeddings+=1
            except:
                if (not word.startswith("#")):
                    word = "#" + word
                tokens=split_hashtag_to_words(word)
                #print('before: ', word, ' after: ', split_hashtag_to_words(word))
                for token in tokens.split():
                    if((len(token) != 1) or (token == "a") or (token == "i")):
                        try:
                            we_tweets[i] += words[token] * word2weight[token]
                            foundEmbeddings+=1
                        except:
                            print('Not found: ', token)
                            continue;
                continue;
        we_tweets[i] /= foundEmbeddings
    return we_tweets
