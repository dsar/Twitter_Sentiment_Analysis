import numpy as np
import pandas as pd
import os
import pickle

from options import *
if options['warnings'] == False:
	pd.options.mode.chained_assignment = None

from utils import *
from preprocessing import *
from baseline import *
from cross_validation import cross_validation
from vectorizer import load_vectorizer
from doc2vec_solution import doc2vec

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier 

import csv

#clear cache
if options['clear']:
	clear_cache(preproc=clear['preproc'],tfidf=clear['tfidf'],pred=clear['pred'], d2v=clear['d2v'])

# Initialization phase
if options['init']:
	print('start init.sh')
	os.system('bash init.sh ' + POS_TWEETS_FILE + ' ' + NEG_TWEETS_FILE)

# Load Data

##Train Data
print('loading data')
pos_tweets = pd.DataFrame(read_file(POS_TWEETS_FILE), columns=['tweet'])
pos_tweets['sentiment'] = 1
neg_tweets = pd.DataFrame(read_file(NEG_TWEETS_FILE), columns=['tweet'])
neg_tweets['sentiment'] = -1
print('positive tweets shape: ',pos_tweets.shape)
print('negative tweets shape: ',neg_tweets.shape)
tweets = pd.concat([pos_tweets, neg_tweets], axis=0)
print('final tweets shape: ',tweets.shape)

##Test Data
test_tweets = pd.DataFrame(read_file(TEST_TWEETS_FILE), columns=['tweet'])
test_tweets['tweet'] = test_tweets.apply(lambda tweet: remove_tweet_id(tweet['tweet']), axis=1)
print('test data shape:', test_tweets.shape)

#Tweets Preprocessing
if options['preprocess'][0]:
	tweets = tweets_preprocessing(tweets,train=True, params=preprocessing_params)
	test_tweets = tweets_preprocessing(test_tweets,train=False, params=preprocessing_params)
	print('processed tweets shape: ',tweets.shape)

# Features extraction
if options['feature_extraction'] == 'WE':
	print('WE')
	if options['we_method'] == 'baseline':
		print('baseline')
		train_reptweets, test_reptweets = baseline(tweets, test_tweets)
		if options['scale']:
			scaler = StandardScaler()
			train_reptweets = scaler.fit_transform(train_reptweets)  
			test_reptweets = scaler.fit_transform(test_reptweets)
	elif options['we_method'] == 'doc2vec':
		print('doc2vec')
		train_reptweets, test_reptweets = doc2vec(tweets, test_tweets)

	# PCA
	if options['PCA'][0]:
	    print('PCA')
	    pca = decomposition.PCA(n_components=options['PCA'][1])
	    train_reptweets = pca.fit_transform(train_reptweets)
	    test_reptweets = pca.transform(test_reptweets)
	#Polynomial expansion
	if  options['poly'][0]:
		print('poly')
		poly = PolynomialFeatures(options['poly'][1])
		train_reptweets = poly.fit_transform(train_reptweets)
		test_reptweets = poly.fit_transform(test_reptweets)

# TFIDF
elif options['feature_extraction'] == 'TFIDF':
	train_reptweets, test_reptweets = load_vectorizer(tweets, test_tweets)

# Apply ML algorithm
if options['ml_algorithm'] == 'RF':
	print('init Random Forest')
	clf = RandomForestClassifier(n_estimators=100,max_depth=50,n_jobs=-1,random_state=4)
elif options['ml_algorithm'] == 'SVM':
	print('init SVM')
	clf = svm.LinearSVC(max_iter=10000)
elif options['ml_algorithm'] == 'LR':
	print('init Logistic Regression')
	clf = linear_model.LogisticRegression(C=1e5,n_jobs=-1,max_iter=10000)
elif options['ml_algorithm'] == 'NN':
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 2), random_state=4, verbose=False)

# perform cv
if options['cv'][0]:
	print('CV')
	avg_test_accuracy, cv = cross_validation(clf, 
											tweets.shape[0],
											train_reptweets,
											tweets['sentiment'], 
											n_folds=options['cv'][1])
	print('cv avg score: ',avg_test_accuracy)

# train selected model
print('training')
clf.fit(train_reptweets, tweets['sentiment'])
pred = clf.predict(test_reptweets)

# CAUTION with tweets ids(!)
print('pred shape: ',pred.shape)
print('pred values: ',pred[0:20])
print(train_reptweets)
# Write predictions to file
print('create final csv submission file')
create_csv_submission(pred)
