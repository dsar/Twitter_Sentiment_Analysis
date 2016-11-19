import nltk
import re
import numpy as np

def filter_user(tweets):
	"""tweets: Series"""
	return tweets.str.replace('<user>', '', case=False)

def filter_url(tweets):
	return tweets.str.replace('<url>', '', case=False)

def filter_hashtag(tweets):
	return tweets.str.replace('^#', '', case=False)

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
