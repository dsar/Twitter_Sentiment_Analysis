# This python file contains functions that help us deal with raw tweets preprocessing

#Import libraries
import nltk
import re
import pandas as pd
import numpy as np
import pickle
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text
from split_hashtag import split_hashtag_to_words
from options import *


# Global Operations

## Initialize Stopwords
stopWords = stopwords.words("english")
## Remove words that denote sentiment
for w in ['no', 'not', 'nor', 'only', 'against', 'up', 'down', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ain', 'aren', 'mightn', 'mustn', 'needn', 'shouldn', 'wasn', 'weren', 'wouldn']:
    stopWords.remove(w)

## Initialize Stemmer
lancaster_stemmer = LancasterStemmer()
## Initialize Lemmatizer
lmt = nltk.stem.WordNetLemmatizer()
## Initialize tokenizer
punc_tokenizer = RegexpTokenizer(r'\w+')

## Build Sentiment Lexicon
positiveWords = set(open(POSITIVE_WORDS, encoding = "ISO-8859-1").read().split())
negativeWords = set(open(NEGATIVE_WORDS, encoding = "ISO-8859-1").read().split())


# Functions

def filter_user(tweets):
    """
    DESCRIPTION: 
            Filters the word '<user>' from a tweet (replace it with empty string)
    INPUT: 
            tweets: Series of a set of tweets as a python strings
    OUTPUT: 
            Series of <user>filtered tweets
    """
    return tweets.str.replace('<user>', '', case=False)

def filter_url(tweets):
    """
    DESCRIPTION: 
            filters the word '<url>' from a tweet (replace it with empty string)
    INPUT: 
            tweets: Series of a set of tweets as a python strings
    OUTPUT: 
            Series of <url>-filtered tweets
    """
    return tweets.str.replace('<url>', '', case=False)

def expand_not(tweets):
    """
    DESCRIPTION: 
            In informal speech, which is widely used in social media, it is common to use contractions of words 
	    (e.g., don't instead of do not).
	    This may result in misinterpreting the meaning of a phrase especially in the case of negations.
            This function expands these contractions and other similar ones (e.g it's --> it is etc...).
    INPUT: 
            tweets: Series of a set of tweets as a python strings
    OUTPUT: 
            Series of filtered tweets
    """
    tweets = tweets.str.replace('n\'t', ' not', case=False)
    tweets = tweets.str.replace('i\'m', 'i am', case=False)
    tweets = tweets.str.replace('\'re', ' are', case=False)
    tweets = tweets.str.replace('it\'s', 'it is', case=False)
    tweets = tweets.str.replace('that\'s', 'that is', case=False)
    tweets = tweets.str.replace('\'ll', ' will', case=False)
    tweets = tweets.str.replace('\'l', ' will', case=False)
    tweets = tweets.str.replace('\'ve', ' have', case=False)
    tweets = tweets.str.replace('\'d', ' would', case=False)
    tweets = tweets.str.replace('he\'s', 'he is', case=False)
    tweets = tweets.str.replace('what\'s', 'what is', case=False)
    tweets = tweets.str.replace('who\'s', 'who is', case=False)
    tweets = tweets.str.replace('\'s', '', case=False)

    for punct in ['!', '?', '.']:
        regex = "(\\"+punct+"( *)){2,}"
        tweets = tweets.str.replace(regex, punct+' <repeat> ', case=False)

    return tweets


def emoji_transformation(tweet):
    """
    DESCRIPTION: 
                transforms emoticons to sentiment tags e.g :) --> <smile>
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            transformed tweet as a python string
    """

    #Construct emojis

    hearts = ["<3", "♥"]
    eyes = ["8",":","=",";"]
    nose = ["'","`","-",r"\\"]
    smilefaces = []
    lolfaces = []
    sadfaces = []
    neutralfaces = []

    for e in eyes:
        for n in nose:
            for s in ["\)", "d", "]", "}"]:
                smilefaces.append(e+n+s)
                smilefaces.append(e+s)
            for s in ["\(", "\[", "{"]:
                sadfaces.append(e+n+s)
                sadfaces.append(e+s)
            for s in ["\|", "\/", r"\\"]:
                neutralfaces.append(e+n+s)
                neutralfaces.append(e+s)
            #reversed
            for s in ["\(", "\[", "{"]:
                smilefaces.append(s+n+e)
                smilefaces.append(s+e)
            for s in ["\)", "\]", "}"]:
                sadfaces.append(s+n+e)
                sadfaces.append(s+e)
            for s in ["\|", "\/", r"\\"]:
                neutralfaces.append(s+n+e)
                neutralfaces.append(s+e) 
            lolfaces.append(e+n+"p")
            lolfaces.append(e+"p")

    smilefaces = set(smilefaces)
    lolfaces = set(lolfaces)
    sadfaces = set(sadfaces)
    neutralfaces = set(neutralfaces)

    t = []
    for w in tweet.split():
        if(w in hearts):
            t.append("<heart>")
        elif(w in smilefaces):
            t.append("<smile>")
        elif(w in lolfaces):
            t.append("<lolface>")
        elif(w in neutralfaces):
            t.append("<neutralface>")
        elif(w in sadfaces):
            t.append("<sadface>")
        else:
            t.append(w)
    return (" ".join(t)).strip()

def emphasize_sentiment_words(tweet):
    """
    DESCRIPTION: 
                By using an opinion lexicon, if a tweet contained a positive or negative word
                 from that lexicon, it is emphasized respectively
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            a transformed tweet as a python string
    """
    t = []
    for w in tweet.split():
        if w in positiveWords:
            t.append('positive ' + w)
        elif w in negativeWords:
            t.append('negative ' + w)
        else:
            t.append(w)
    return (" ".join(t)).strip()

def split_hashtag(tweet):
    """
    DESCRIPTION: 
                Applies segmentation in a tweet's hashtag e.g "#thankyou" -> "thank you"
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            tranformed tweet with splitted hashtag
    """
    t = []
    for w in tweet.split():
        if w.startswith("#"):
            t.append("<hashtag>")
        t.append(w)
    return (" ".join(t)).strip()

def filter_digits(tweet):
    """
    DESCRIPTION: 
                Filters digits from a tweet. Words that contain digits are not filtered.
    INPUT: 
                tweet: a tweet as a python string
    OUTPUT: 
                digit-filtered tweet as a python string
    """
    t = []
    for w in tweet.split():
        try:
            num = re.sub('[,\.:%_\-\+\*\/\%\_]', '', w)
            float(num)
            t.append("<number>")
        except:
            t.append(w)
    return (" ".join(t)).strip()
 
def filter_small_words(tweet):
    """
    DESCRIPTION: 
                filters words of one character from a tweet (replace them with an empty string)
    INPUT: 
                tweet: a tweet as a python string
    OUTPUT: 
                small words-filtered tweet as a python string
    """
    return " ".join([w for w in tweet.split() if len(w) >1 or not w.isalpha()])


def tokenization(tweet):
    """
    DESCRIPTION: 
                Tokenizes a tweet into words
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            list of tweet's tokens (words)
    """
    return list(tweet.split())

def pos_tag(tweet):
    """
    DESCRIPTION: 
                assigns part of speech tags to each word of a tweet
                e.g pos_tag('the dog is black')
                    returns: [('the', 'DT'), ('dog', 'NN'), ('is', 'VBZ'), ('black', 'JJ')]
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            list of (word,tag) pairs for each word of the tweet
    """
    tweet = nltk.word_tokenize(tweet.lower())
    return nltk.pos_tag(tweet)

def filter_repeated_chars_on_tweet(tweet):
    """
    DESCRIPTION: 
                filters repeated characters from each word of a given tweet
                e.g "I am haaaaaaaaaaaappy" returns: "I am haappy"
                Thus, we different words with exactly the same meaning are equalized
                e.g haaaaaha & haaaaaaaaaaaaaaaaaaha are equal.
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            transformed tweet as a python string
    """
    t = []
    for w in tweet.split():
        t.append(re.sub(r'([a-z])\1\1+$', r'\1 <elong>', w))
    return (" ".join(t)).strip()

def remove_tweet_id(tweet):
    """
    DESCRIPTION: 
                removes the id from a string that contains an id and a tweet
                e.g "<id>,<tweet>" returns "<tweet>"
    INPUT: 
            tweet: a python string which contains an id concatinated with a tweet of the following format:
           "<id>,<tweet>"
    OUTPUT: 
            only the tweet is returned as a python string
    """
    return tweet.split(',', 1)[-1]

def convert_to_lowercase(tweets):
	"""
    DESCRIPTION: 
                Applies to lowercase to a pandas Series of tweets
    INPUT: 
            tweets: a Series of tweets as strings
    OUTPUT: 
            a Series of to lower case tweets as python strings
    """
	return tweets.str.lower()

def topk_most_important_features(vectorizer, clf, class_labels=['pos','neg'],k=10):    
    """
    DESCRIPTION: 
                Returns features with the highest coefficient values, per class label
    INPUT: 
           vectorizer: a fitted vectorizer (like tfidf_vectorizer or count_vectorizer)
           clf: a fitted sklearn classifier
           class_labels: a list of all the possible class labels (in our case pos, neg)
           k: top k valuable features
    OUTPUT: 
           top k most valueable features
    """
    important_features = []
    feature_names = vectorizer.get_feature_names()
    topk = np.argsort(clf.coef_[0])[-k:]
    for j in topk:
        important_features.append(feature_names[j])	
    return important_features[::-1]

def show_most_informative_features(vectorizer, clf, n=20):
    """
    DESCRIPTION: returns features with the highest coefficient values, per class label
    INPUT: vectorizer: a fitted vectorizer (like tfidf_vectorizer or count_vectorizer)
           clf: a fitted sklearn classifier
           k: top k valuable features
    OUTPUT: 
           top k most valueable features
    """
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

def lemmatize_single(w):
    """
    DESCRIPTION: 
                Lemmatize a single word
    INPUT:  
            w: a word as a python string
    OUTPUT: 
            lemmatized word as a python string. In case the word cannot be lemmatized
            it will be returned in its first form.
    """
    try:
        a = lmt.lemmatize(w).lower()
        return a
    except Exception as e:
        return w

def lemmatize(tweet):
    """
    DESCRIPTION: 
                Lemmatize all words from a tweet one by one
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            Lemmatized tweet as a python string.
    """
    x = [lemmatize_single(t) for t in tweet.split()]
    return " ".join(x)

def stemming_single(word):
    """
    DESCRIPTION: 
                Apply stemming to a single word
    INPUT: 
            w: a word as a python string
    OUTPUT: 
            stemmed word as a python string. In case the word cannot be lemmatized
            it will be returned in its first form.
    """
    return lancaster_stemmer.stem(word)    

def stemming(tweet):
    """
    DESCRIPTION: 
                Apply stemming to all the words from a tweet one by one
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            stemmed tweet as a python string.
    """
    x = [stemming_single(t) for t in tweet.split()]
    return " ".join(x)

def filter_punctuation(tweet):
    """
    DESCRIPTION: 
                Filters punctuation from a tweet
    INPUT: 
            tweet: a tweet as a python string
    OUTPUT: 
            punctuation-filtered tweet as a python string.
    """
    return " ".join(punc_tokenizer.tokenize(tweet))

class LemmaTokenizer(object):
    """
    DESCRIPTION: 
                LemmaTokenizer class to be used in Vectorizer
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def cache_preprocessing(tweets, train=True):
    """
    DESCRIPTION: 
                Caches preprocessed tweets from a previous execution and writes them to a csv file
    INPUT: 
            tweets: pandas DataFrame containing all the tweets
            train: boolean var e.g train=True for train tweets, train=False for test tweets
    OUTPUT: 
            -
    """
    from pathlib import Path    
    Path(PREPROC_DATA_PATH ).mkdir(exist_ok=True)
    if train:
        tweets.to_csv(path_or_buf=TRAIN_PREPROC_CACHING_PATH, sep=',' , header=False ,encoding='utf-8' ,index=False)
    else:
        tweets.to_csv(path_or_buf=TEST_PREPROC_CACHING_PATH, sep=',', header=False ,encoding='utf-8' ,index=False)

def load_preprocessed_tweets(train=True):
    """
    DESCRIPTION: 
                Loads preprocessed tweets from a previous execution
    INPUT:
            train: boolean var e.g train=True for train tweets, train=False for test tweets
    OUTPUT: 
            a DataFrame with the loaded tweets or None if the load fails
            True/False depending on the laod status
    """
    from pathlib import Path
    if train:
        path = TRAIN_PREPROC_CACHING_PATH
    else:
        path = TEST_PREPROC_CACHING_PATH
    my_file = Path(path)
    if my_file.is_file():
        return pd.read_csv(path,sep=',', names=['tweet','sentiment'],encoding='utf-8'), True
    print('\nThere is no cached file for preprocessed tweets\n')
    return None, False

def tweets_preprocessing(tweets, train=True, params=None):
    """
    DESCRIPTION: preprocesses a dataframe of a set of tweets given a set of parameters
    INPUT:
            tweets: a Dataframe which contains a set of arbitrary tweets
            train: boolean var e.g train=True for train tweets, train=False for test tweets
            params: a dictionary of parameters for preprocessing actions
    OUTPUT: 
            a DataFrame with the loaded tweets or None if the load fails
            True/False depending on the laod status
    """  
    if params == None:
        print('set default parameters')
        fduplicates = frepeated_chars = fpunctuation = fuser = furl = fhashtag = fdigits = fsmall_words = fstopwords = transform_emojis = sentiment_words = True
    else:
        fduplicates = params['fduplicates']
        frepeated_chars = params['frepeated_chars']
        fexpand_not = params['fexpand_not']
        fpunctuation = params['fpunctuation']
        fuser = params['fuser']
        furl = params['furl']
        fhashtag = params['fhashtag']
        fdigits = params['fdigits']
        fsmall_words = params['fsmall_words']
        fstopwords = params['fstopwords']
        transform_emojis = params['transform_emojis']
        sentiment_words = params['sentiment_words']

        if train:
            print_dict_settings(params,msg='Preprocessing Settings:\n')

    if train:    
        print('Tweets Preprocessing for the Training set started\n')
    else:
        print('Tweets Preprocessing for the Testing set started\n')

    stored_tweets, read = load_preprocessed_tweets(train)
    if read:
        print('\nTweets have been successfully loaded!')
        return stored_tweets

    if train:
        if fduplicates:
            print('Number of tweets before duplicates removal:\t', tweets.shape[0])
            tweets = tweets.drop_duplicates(subset='tweet')
            print('Number of tweets after duplicates removal:\t', tweets.shape[0])
            print('Duplicates removal DONE')

    if frepeated_chars:
        tweets['tweet'] = tweets.apply(lambda tweet: filter_repeated_chars_on_tweet(tweet['tweet']), axis=1)
        print('Repeated characters filtering DONE')

    if sentiment_words:
        tweets['tweet'] = tweets.apply(lambda tweet: emphasize_sentiment_words(tweet['tweet']), axis=1)
        print('Sentiment words emphasizing DONE')

    if transform_emojis:
        tweets['tweet'] = tweets.apply(lambda tweet: emoji_transformation(tweet['tweet']), axis=1)
        print('Transforming emojis DONE')

    if fexpand_not:
        tweets['tweet'] = expand_not(tweets['tweet'])
        print('Expanding not DONE')

    if fpunctuation:
        tweets['tweet'] = tweets.apply(lambda tweet: filter_punctuation(tweet['tweet']), axis=1)
        print('Punctuation filtering DONE')

    if fuser:
        tweets['tweet'] = filter_user(tweets['tweet'])
        print('User filtering DONE')
    
    if furl:
        tweets['tweet'] = filter_url(tweets['tweet'])
        print('Url filtering DONE')
    
    if fhashtag:
        tweets['tweet'] = tweets.apply(lambda tweet: split_hashtag(tweet['tweet']), axis=1)
        print('Hashtag splitting DONE')
    
    if fdigits:
        tweets['tweet'] = tweets.apply(lambda tweet: filter_digits(tweet['tweet']), axis=1)
        print('Digits DONE')
    
    if fsmall_words:
        tweets['tweet'] = tweets.apply(lambda tweet: filter_small_words(tweet['tweet']), axis=1)
        print('Small words filtering DONE')

    if fstopwords:
        tweets['tweet'] = tweets.apply(lambda tweet: remove_stopwords_from_tweet(tweet['tweet']), axis=1)
        print('Stopwords filtering DONE')

    if algorithm['options']['preprocess'][1] == 'save':
        print('\nSaving preprocessed tweets...')
        cache_preprocessing(tweets, train=train)
        print('DONE')
    else:
        print('\nPreprocessed tweets did not saved...')

    print('Tweets Preprocessing have been successfully finished!\n')

    return tweets


def remove_stopwords_from_tweet(tweet):
    """
    DESCRIPTION: filters stopwords from a tweet
    INPUT:
            tweet: a tweet as a python string
    OUTPUT: 
            a stopword-filtered tweet as a python string
    """
    tokens = tweet.split()
    for word in tokens:
        if word in stopWords:
            tokens.remove(word)
    return ' '.join(tokens)

