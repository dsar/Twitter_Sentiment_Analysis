import numpy as np
import pandas as pd
import os

from options import *
if options['warnings'] == False:
	pd.options.mode.chained_assignment = None

from utils import *
from plots import *
from preprocessing import *
from cross_validation import *
from baseline import *


from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# Initialization phase
# 5 words elim problem!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if options['init']:
	print('start init.sh')
	os.system('bash init.sh')

# Load Data

#Train Data
print('loading data')
pos_tweets = pd.read_table(DATA_PATH+POS_TWEETS_FILE, names=['tweet','sentiment'])
pos_tweets['sentiment'] = 'pos'
neg_tweets = pd.read_table(DATA_PATH+NEG_TWEETS_FILE ,names=['tweet','sentiment'])
neg_tweets['sentiment'] = 'neg'
print('positive tweets shape: ',pos_tweets.shape)
print('negative tweets shape: ',neg_tweets.shape)
tweets = pd.concat([pos_tweets, neg_tweets], axis=0)
print('final tweets shape: ',tweets.shape)

#Test Data
test_tweets = pd.read_table(DATA_PATH+TEST_TWEETS_FILE, names=['tweet','sentiment'])
test_tweets['tweet'] = test_tweets.apply(lambda tweet: remove_tweet_id(tweet['tweet']), axis=1)
print('test data shape:', test_tweets.shape)

#Tweets Preprocessing
if options['preprocess']:
	tweets = preprocessing(tweets,train=True, params=preprocessing_params)
	test_tweets = preprocessing(test_tweets,train=False, params=preprocessing_params)

# Features extraction
we_tweets, we_test_tweets = baseline(tweets, test_tweets)

# Apply algorithm
print('Random Forest')
forest = RandomForestClassifier(n_estimators=10,max_depth=10,n_jobs=-1,random_state=4)
forest.fit(we_tweets, tweets['sentiment'])
we_test_tweets = np.nan_to_num(we_test_tweets)  #!!!!!!! under discussion
pred = forest.predict(we_test_tweets)
print('pred shape: ',pred.shape)
print('pred values: ',pred)

# Write predictions to file
print('create final csv submission file')
create_csv_submission(pred)
