import numpy as np
import pandas as pd
import os

from options import *
if options['warnings'] == False:
	pd.options.mode.chained_assignment = None

from utils import *
from preprocessing import *
from baseline import *
from cross_validation import cross_validation

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

# Initialization phase
# 5 words elim problem!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if options['init']:
	print('start init.sh')
	os.system('bash init.sh')

# Load Data

#Train Data
print('loading data')
pos_tweets = pd.read_table(DATA_PATH+POS_TWEETS_FILE, names=['tweet','sentiment'])
pos_tweets['sentiment'] = 1
neg_tweets = pd.read_table(DATA_PATH+NEG_TWEETS_FILE ,names=['tweet','sentiment'])
neg_tweets['sentiment'] = -1
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

# Apply ML algorithm
if options['ml_algorithm'] == 'RF':
	print('init Random Forest')
	clf = RandomForestClassifier(n_estimators=20,max_depth=25,n_jobs=-1,random_state=4)
elif options['ml_algorithm'] == 'SVM':
	print('init SVM')
	clf = svm.LinearSVC(max_iter=500)
elif options['ml_algorithm'] == 'LR':
	print('init Logistic Regression')
	clf = linear_model.LogisticRegression(C=1e5)

# perform cv
if options['cv']:
	print('CV')
	avg_test_accuracy, cv = cross_validation(clf, tweets.shape[0], we_tweets, tweets['sentiment'], n_folds=options['k_fold'])
	print('cv avg score: ',avg_test_accuracy)

# train selected model
print('training')
clf.fit(we_tweets, tweets['sentiment'])
pred = clf.predict(we_test_tweets)

# CAUTION with tweets ids(!)
print('pred shape: ',pred.shape)
print('pred values: ',pred[0:20])

# Write predictions to file
print('create final csv submission file')
create_csv_submission(pred)
