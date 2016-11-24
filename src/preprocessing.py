# This python file contains functions that help us deal with raw tweets preprocessing

#Import libraries
import nltk
import re
import pandas as pd
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction import text
from options import preprocessing_params, print_dict_settings, \
                    DATA_PATH, \
                    TRAIN_PREPROC_CACHING_PATH, TEST_PREPROC_CACHING_PATH


# Initialization
lancaster_stemmer = LancasterStemmer()
lmt = nltk.stem.WordNetLemmatizer()
punc_tokenizer = RegexpTokenizer(r'\w+')

def filter_user(tweets):
	"""tweets: Series"""
	return tweets.str.replace('user', '', case=False)

def filter_url(tweets):
	return tweets.str.replace('url', '', case=False)

def filter_hashtag(tweets):
	return tweets.str.replace('^#', '', case=False)

def filter_digits(tweet):
	from string import digits
	import re
	remove_digits = str.maketrans('', '', digits)
	return re.sub(' +',' ',tweet.translate(remove_digits))
 
def filter_small_words(tweet):
	return " ".join([w for w in tweet.split() if len(w) >2])


def tokenization(tweet):
    return list(tweet.split())

def pos_tag(tweet):
    tweet = nltk.word_tokenize(tweet.lower())
    return nltk.pos_tag(tweet)

def filter_repeated_chars_on_tweet(tweet):
	return re.sub(r'(.)\1+', r'\1\1', tweet)

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

def cache_preprocessing(tweets):
    tweets.to_csv(path_or_buf=DATA_PATH+TRAIN_PREPROC_CACHING_PATH, sep=',', encoding='utf-8' ,index=False)

def load_preprocessed_tweets():
    from pathlib import Path
    my_file = Path(DATA_PATH+TRAIN_PREPROC_CACHING_PATH)
    if my_file.is_file():
        return pd.read_csv(DATA_PATH+TRAIN_PREPROC_CACHING_PATH,sep=',',encoding='utf-8'), True
    print('\nThere is no cached file for preprocessed tweets\n')
    return None, False

def preprocessing(tweets, train=True, params=None):
    """
    -Duplicates are removed to avoid putting extra weight on any particular tweet.
    -We use preprocessing so that any letter occurring more than two times in a row is replaced with two occurrences.
     As an example, the words haaaaaaaaappy and haaaaappy should be converted to haappy
    """

    if params == None:
        print('set default parameters')
        fduplicates = frepeated_chars = fpunctuation = fuser = furl = fhashtag = fdigits = fsmall_words = save = True
    else:
        fduplicates = params['fduplicates']
        frepeated_chars = params['frepeated_chars']
        fpunctuation = params['fpunctuation']
        fuser = params['fuser']
        furl = params['furl']
        fhashtag = params['fhashtag']
        fdigits = params['fdigits']
        fsmall_words = params['fsmall_words']
        save = params['save']

        print_dict_settings(params,msg='Preprocessing Settings:\n')

    if train:    
        print('Tweets Preprocessing for the Training set started\n')
    else:
        print('Tweets Preprocessing for the Testing set started\n')

    stored_tweets, read = load_preprocessed_tweets()
    if train==True and read == True:
        print('\nTweets have been successfully loaded!')
        stored_tweets['tweet'] = stored_tweets['tweet'].fillna('the')  #!!!!!!! under discussion
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
        tweets['tweet'] = filter_hashtag(tweets['tweet'])
        print('Hashtag filtering DONE')
    
    if fdigits:
        tweets['tweet'] = tweets.apply(lambda tweet: filter_digits(tweet['tweet']), axis=1)
        print('Digits DONE')
    
    if fsmall_words:
        tweets['tweet'] = tweets.apply(lambda tweet: filter_small_words(tweet['tweet']), axis=1)
        print('Small words filtering DONE')
        
    if train and save:
        print('\nSaving preprocessed tweets...')
        cache_preprocessing(tweets)
        print('DONE')
    else:
        print('\nPreprocessed tweets did not saved...')

    print('Tweets Preprocessing have been successfully finished!\n')

    return tweets

def find_stopwords(number_of_stopwords=100):
    stoplist = stopwords.words('english')
    fdist = FreqDist(stoplist)
    top = fdist.most_common(number_of_stopwords)
    top = [x[0] for x in top]
    # In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stop_words = set(top)
    my_stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
    return my_stop_words
