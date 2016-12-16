import numpy as np
import pandas as pd

import os
import pickle
import csv
import itertools

from utils import *
from preprocessing import tweets_preprocessing, remove_tweet_id
from baseline import baseline
from cross_validation import cross_validation
from vectorizer import load_vectorizer
from tfidf_embdedding_vectorizer import tfidf_embdedding_vectorizer
from doc2vec_solution import doc2vec
from fast_text import fast_text
from options import *
if options['warnings'] == False:
	pd.options.mode.chained_assignment = None

from sklearn import svm, linear_model, preprocessing, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPClassifier 

from trainCNN import trainCNN
from evalCNN import evalCNN

#clear cache
if options['clear']:
	clear_cache(clear_params)

# Load Data
print('Loading data')

pos_tweets = pd.DataFrame(read_file(POS_TWEETS_FILE), columns=['tweet'])
pos_tweets['sentiment'] = 1
neg_tweets = pd.DataFrame(read_file(NEG_TWEETS_FILE), columns=['tweet'])
neg_tweets['sentiment'] = -1
tweets = pd.concat([pos_tweets, neg_tweets], axis=0)

test_tweets = pd.DataFrame(read_file(TEST_TWEETS_FILE), columns=['tweet'])
test_tweets['tweet'] = test_tweets.apply(lambda tweet: remove_tweet_id(tweet['tweet']), axis=1)

#Data Shape
print('\tpositive tweets shape: ',pos_tweets.shape)
print('\tnegative tweets shape: ',neg_tweets.shape)
print('\tfinal tweets shape: ',tweets.shape)
print('\ttest data shape:', test_tweets.shape)

#Tweets Preprocessing
if options['preprocess'][0]:
	tweets = tweets_preprocessing(tweets,train=True, params=preprocessing_params)
	test_tweets = tweets_preprocessing(test_tweets,train=False, params=preprocessing_params)

## FastText case
if options['ml_algorithm'] == 'FT':
	print('\nInitializing FastText')
	pred = fast_text(tweets, test_tweets)
	# Preview of the results
	print('pred shape: ',len(pred))
	print('pred values: ',pred[0:20])
	print('Creating final csv submission file')
	create_csv_submission(pred)
	exit()

# Features extraction
if options['feature_extraction'] == 'WE':
	print_dict_settings(WE_params, msg='\nWord Embeddings Parameters:')
	print('Feature extraction using WE\n')
	if options['we_method'] == 'we_mean':
		print('\tUsing WE mean')
		train_reptweets, test_reptweets = baseline(tweets, test_tweets)
	elif (options['we_method'] == 'dm_doc2vec') or (options['we_method'] == 'dbow_doc2vec'):
		print('\tUsing doc2vec')
		train_reptweets, test_reptweets = doc2vec(tweets, test_tweets)
	elif options['we_method'] == 'we_tfidf':
		print('\tUsing WE tfidf')
		train_reptweets, test_reptweets = tfidf_embdedding_vectorizer(tweets, test_tweets)

	# Scale matrices
	if options['scale']:
		print('Scaling Matrices\n')
		scaler = StandardScaler()
		train_reptweets = scaler.fit_transform(train_reptweets)
		scaler = StandardScaler()  
		test_reptweets = scaler.fit_transform(test_reptweets)

	# Apply PCA
	if options['PCA'][0]:
	    print('Appling PCA with '+str(options['PCA'][1])+' number of components')
	    pca = decomposition.PCA(n_components=options['PCA'][1])
	    train_reptweets = pca.fit_transform(train_reptweets)
	    pca = decomposition.PCA(n_components=options['PCA'][1])
	    test_reptweets = pca.fit_transform(test_reptweets)

	#Polynomial expansion
	if  options['poly'][0]:
		print('Polynomial expansion with '+str(options['poly'][1])+' base')
		poly = PolynomialFeatures(options['poly'][1])
		train_reptweets = poly.fit_transform(train_reptweets)
		poly = PolynomialFeatures(options['poly'][1])
		test_reptweets = poly.fit_transform(test_reptweets)

elif options['feature_extraction'] == 'TFIDF':
	print('Feature extraction using TF-IDF')
	train_reptweets, test_reptweets = load_vectorizer(tweets, test_tweets)

if options['model_selection']:
	# param init
	if options['ml_algorithm'] == 'RF':
		print('Initializing Random Forest')
		clf = RandomForestClassifier(n_estimators=100,max_depth=50,n_jobs=-1,random_state=4)
	elif options['ml_algorithm'] == 'SVM':
		print('SVM params init')
		listOLists = [['hinge','squared_hinge'],[0.1,0.5,0.8,1,2]]
		clf = svm.LinearSVC(max_iter=10000)
	elif options['ml_algorithm'] == 'LR':
		print('Initializing Logistic Regression')
		clf = linear_model.LogisticRegression(C=1e5,n_jobs=-1,max_iter=10000)
	elif options['ml_algorithm'] == 'NN':
		print('NN params init')
		listOLists = [[1e-2, 1e-3, 1e-4, 1e-5],['constant', 'invscaling', 'adaptive'],['lbfgs', 'sgd', 'adam'],[1,2],[8,16,32,64,128]]
	c = itertools.product(*listOLists)
	print('combinations: ',c)

	max_tuple = []
	max_avg_score = 0
	for tuple_ in c:
		print('tuple: ',tuple_)
		# Apply ML algorithm
		if options['ml_algorithm'] == 'RF':
			print('Initializing Random Forest')
			clf = RandomForestClassifier(n_estimators=100,max_depth=50,n_jobs=-1,random_state=4)
		elif options['ml_algorithm'] == 'SVM':
			print('Initializing SVM')
			clf = svm.LinearSVC(max_iter=10000,intercept_scaling=tuple_[1],loss=tuple_[0])
		elif options['ml_algorithm'] == 'LR':
			print('Initializing Logistic Regression')
			clf = linear_model.LogisticRegression(C=1e5,n_jobs=-1,max_iter=10000)
		elif options['ml_algorithm'] == 'NN':
			print('Initializing Neural Network')
			clf = MLPClassifier(solver=tuple_[2],\
								activation='logistic', \
								hidden_layer_sizes=(tuple_[4], tuple_[3]), \
								random_state=4, \
								verbose=False,\
							    alpha=tuple_[0],\
								learning_rate = tuple_[1])

		# Cross Validation
		if options['cv'][0]:
			print('Cross-validating results')
			avg_test_accuracy, cv = cross_validation(clf, 
													tweets.shape[0],
													train_reptweets,
													tweets['sentiment'], 
													n_folds=options['cv'][1])
			print('Avg CV score: ',avg_test_accuracy)

		if avg_test_accuracy > max_avg_score:
			max_tuple = tuple_
			max_avg_score = avg_test_accuracy

	print('max_avg_score', max_avg_score)
	print('max_tuple', max_tuple)


## best parameters hardcoded
else:
	# Apply ML algorithm
	if options['ml_algorithm'] == 'RF':
		print('\nInitializing Random Forest')
		clf = RandomForestClassifier(n_estimators=100,max_depth=50,n_jobs=-1,random_state=4)
	elif options['ml_algorithm'] == 'SVM':
		print('\nInitializing SVM')
		clf = svm.LinearSVC(max_iter=SVM['max_iter'], intercept_scaling=SVM['intercept_scaling'], loss=SVM['loss'])
	elif options['ml_algorithm'] == 'LR':
		print('\nInitializing Logistic Regression')
		clf = linear_model.LogisticRegression(C=LR['C'],max_iter=LR['max_iter'],n_jobs=-1)
	elif options['ml_algorithm'] == 'NN':
		print('\nInitializing Neural Network')
		clf = MLPClassifier(solver=NN['solver'], activation=NN['activation'], hidden_layer_sizes=(NN['k'],NN['hidden_layers']),\
							 random_state=4, verbose=False, max_iter=NN['max_iter'], tol=NN['tol'])
	elif options['ml_algorithm'] == 'CNN':
		labels = np.zeros((tweets.shape[0], 2))
		labels[pos_tweets.shape[0]:, 0] = 1.0
		labels[:pos_tweets.shape[0], 1] = 1.0
		path = trainCNN(tweets, labels)
		pred = evalCNN(test_tweets, path)
		create_csv_submission(pred)
		exit()

	# Cross Validation
	if options['cv'][0]:
		print('Cross Validation...')
		avg_test_accuracy, cv = cross_validation(clf, 
												tweets.shape[0],
												train_reptweets,
												tweets['sentiment'], 
												n_folds=options['cv'][1])
		print('Avg CV score: ',avg_test_accuracy)


# Train model
print('Training model')
clf.fit(train_reptweets, tweets['sentiment'])
print('Predicting')
pred = clf.predict(test_reptweets)

# Preview of the results
print('pred shape: ',pred.shape)
print('pred values: ',pred[0:20])
# Write predictions to file
print('Creating final csv submission file')
create_csv_submission(pred)
