import nltk
import re

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