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
from sklearn.feature_extraction import text
from split_hashtag import split_hashtag_to_words
from options import preprocessing_params, print_dict_settings, \
                    TRAIN_PREPROC_CACHING_PATH, TEST_PREPROC_CACHING_PATH, \
                    options


# Initialization
stopWords = stopwords.words("english")
#remove words that denote sentiment
for w in ['no', 'not', 'nor', 'only', 'against', 'up', 'down', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ain', 'aren', 'mightn', 'mustn', 'needn', 'shouldn', 'wasn', 'weren', 'wouldn']:
    stopWords.remove(w)

lancaster_stemmer = LancasterStemmer()
lmt = nltk.stem.WordNetLemmatizer()
punc_tokenizer = RegexpTokenizer(r'\w+')

def filter_user(tweets):
	return tweets.str.replace('<user>', '', case=False)

def filter_url(tweets):
	return tweets.str.replace('<url>', '', case=False)

def expand_not(tweets):
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
            for s in ["\)", "d", "\]", "}"]:
                sadfaces.append(s+n+e)
                sadfaces.append(s+e)
            for s in ["\|", "\/", r"\\"]:
                neutralfaces.append(s+n+e)
                neutralfaces.append(s+e) 
            lolfaces.append(e+n+"p")
            lolfaces.append(e+"p")

    smilefaces.extend([">:d","<[^_^]>"])
    sadfaces.extend(["0_o","0_0","0-0","0_0","0__0","0___0","0,0","0.0"])

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




def split_hashtag(tweet):
    t = []
    for w in tweet.split():
        if w.startswith("#"):
            t.append("<hashtag>")
        t.append(w)
    return (" ".join(t)).strip()

def filter_digits(tweet):
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
	return " ".join([w for w in tweet.split() if len(w) >1 or not w.isalpha()])


def tokenization(tweet):
    return list(tweet.split())

def pos_tag(tweet):
    tweet = nltk.word_tokenize(tweet.lower())
    return nltk.pos_tag(tweet)

def filter_repeated_chars_on_tweet(tweet):
    t = []
    for w in tweet.split():
        t.append(re.sub(r'([a-z])\1\1+$', r'\1 <elong>', w))
    return (" ".join(t)).strip()

def remove_tweet_id(tweet):
    return tweet.split(',', 1)[-1]

def convert_to_lowercase(tweets):
	"""tweets: Series"""
	return tweets.str.lower()

def topk_most_important_features(vectorizer, clf, class_labels=['pos','neg'],k=10):
    """returns features with the highest coefficient values, per class"""
    important_features = []
    feature_names = vectorizer.get_feature_names()
    topk = np.argsort(clf.coef_[0])[-k:]
    for j in topk:
    	important_features.append(feature_names[j])	
    return important_features[::-1]

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

def lemmatize_single(w):
    try:
        a = lmt.lemmatize(w).lower()
        return a
    except Exception as e:
        return w

def lemmatize(tweet):
    x = [lemmatize_single(t) for t in tweet.split()]
    return " ".join(x)

def stemming_single(word):
    return lancaster_stemmer.stem(word)    

def stemming(tweet):
    x = [stemming_single(t) for t in tweet.split()]
    return " ".join(x)

def filter_punctuation(tweet):
	return " ".join(punc_tokenizer.tokenize(tweet))

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def cache_preprocessing(tweets, train=True):
    if train:
        tweets.to_csv(path_or_buf=TRAIN_PREPROC_CACHING_PATH, sep=',' , header=False ,encoding='utf-8' ,index=False)
    else:
        tweets.to_csv(path_or_buf=TEST_PREPROC_CACHING_PATH, sep=',', header=False ,encoding='utf-8' ,index=False)

def load_preprocessed_tweets(train=True):
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
    -Duplicates are removed to avoid putting extra weight on any particular tweet.
    -We use preprocessing so that any letter occurring more than two times in a row is replaced with two occurrences.
     As an example, the words haaaaaaaaappy and haaaaappy should be converted to haappy
    """

    if params == None:
        print('set default parameters')
        fduplicates = frepeated_chars = fpunctuation = fuser = furl = fhashtag = fdigits = fsmall_words = fstopwords = transform_emojis = True
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

        print_dict_settings(params,msg='Preprocessing Settings:\n')

    if train:    
        print('Tweets Preprocessing for the Training set started\n')
    else:
        print('Tweets Preprocessing for the Testing set started\n')

    stored_tweets, read = load_preprocessed_tweets(train)
    if read:
        print('\nTweets have been successfully loaded!')
        # stored_tweets['tweet'] = stored_tweets['tweet'].fillna('the')  #!!!!!!! under discussion
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

    if options['preprocess'][1] == 'save':
        print('\nSaving preprocessed tweets...')
        cache_preprocessing(tweets, train=train)
        print('DONE')
    else:
        print('\nPreprocessed tweets did not saved...')

    print('Tweets Preprocessing have been successfully finished!\n')

    return tweets


def remove_stopwords_from_tweet(tweet):
    tokens = tweet.split()
    for word in tokens:
        if word in stopWords:
            tokens.remove(word)
    return ' '.join(tokens)

