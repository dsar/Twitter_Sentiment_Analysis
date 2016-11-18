import nltk

def filter_user(tweets):
	"""tweets: Series"""
	return tweets.str.replace('<user>', '@user', case=False)

def filter_url(tweets):
	return tweets.str.replace('<url>', '@url', case=False)

def filter_hashtag(tweets):
	return tweets.str.replace('^#', '#hashtag', case=False)

def tokenization(tweet):
    return list(tweet.split())

def pos_tag(tweet):
    tweet = nltk.word_tokenize(tweet.lower())
    return nltk.pos_tag(tweet)