import nltk
import re
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer


# Initialization
lancaster_stemmer = LancasterStemmer()
lmt = nltk.stem.WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def filter_user(tweets):
	"""tweets: Series"""
	return tweets.str.replace('user', '', case=False)

def filter_url(tweets):
	return tweets.str.replace('url', '', case=False)

def filter_hashtag(tweets):
	return tweets.str.replace('^#', '', case=False)

def filter_digits(tweet):
	# t = []
	# for w in tweet.split():
	# 	t.append(''.join([i for i in w if not i.isdigit()]))
	# return " ".join(t)
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
	return " ".join(tokenizer.tokenize(tweet))