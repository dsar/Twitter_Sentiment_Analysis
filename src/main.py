import numpy as np
import pandas as pd

import os
import pickle
import csv
import itertools

from options import *
if algorithm['options']['warnings'] == False:
	pd.options.mode.chained_assignment = None
from utils import *
from preprocessing import tweets_preprocessing, remove_tweet_id
from we_mean import we_mean
from cross_validation import cross_validation
from vectorizer import load_vectorizer
from tfidf_embdedding_vectorizer import tfidf_embdedding_vectorizer
from doc2vec_solution import doc2vec
from fast_text import fast_text

from sklearn import svm, linear_model, preprocessing, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import MultinomialNB

from evalCNN import evalCNN
from trainCNN import trainCNN, trainCNN_fromcheckpoint

# Project Stucture initialization
os.system('./create_structure.sh')

# Clear cache
if algorithm['options']['clear']:
	clear_cache()

# Load Data
print('Loading data')

pos_tweets = pd.DataFrame(read_file(POS_TWEETS_FILE), columns=['tweet'])
pos_tweets['sentiment'] = 1
neg_tweets = pd.DataFrame(read_file(NEG_TWEETS_FILE), columns=['tweet'])
neg_tweets['sentiment'] = -1
tweets = pd.concat([pos_tweets, neg_tweets], axis=0)

test_tweets = pd.DataFrame(read_file(TEST_TWEETS_FILE), columns=['tweet'])
test_tweets['tweet'] = test_tweets.apply(lambda tweet: remove_tweet_id(tweet['tweet']), axis=1)

# Data Shape
print('\tpositive tweets shape: ',pos_tweets.shape)
print('\tnegative tweets shape: ',neg_tweets.shape)
print('\tfinal tweets shape: ',tweets.shape)
print('\ttest data shape:', test_tweets.shape)

# Tweets Preprocessing
if algorithm['options']['preprocess'][0]:
	tweets = tweets_preprocessing(tweets,train=True, params=algorithm['options']['preprocessing_params'])
	test_tweets = tweets_preprocessing(test_tweets,train=False, params=algorithm['options']['preprocessing_params'])

# Feature extraction
if 'feature_extraction' in algorithm['options']:
	if algorithm['options']['feature_extraction'] == 'WE':
		print_dict_settings(algorithm['options']['WE'], msg='\nWord Embeddings Parameters:')
		print('Feature extraction using WE\n')

		if algorithm['options']['WE']['tweet2vec_method'] == 'we_mean':
			print('\tUsing WE mean')
			train_reptweets, test_reptweets = we_mean(tweets, test_tweets)
		elif algorithm['options']['WE']['tweet2vec_method'] == 'we_tfidf':
			print('\tUsing WE tfidf')
			train_reptweets, test_reptweets = tfidf_embdedding_vectorizer(tweets, test_tweets)

		# Scale matrices
		if algorithm['options']['scale']:
			print('Scaling Matrices\n')
			scaler = StandardScaler()
			train_reptweets = scaler.fit_transform(train_reptweets)
			scaler = StandardScaler()  
			test_reptweets = scaler.fit_transform(test_reptweets)

		# Apply PCA
		if algorithm['options']['PCA'][0]:
			print('\nApplying PCA with '+str(algorithm['options']['PCA'][1])+' number of components')
			pca = decomposition.PCA(n_components=algorithm['options']['PCA'][1])
			train_reptweets = pca.fit_transform(train_reptweets)
			pca = decomposition.PCA(n_components=algorithm['options']['PCA'][1])
			test_reptweets = pca.fit_transform(test_reptweets)

		#Polynomial expansion
		if algorithm['options']['poly'][0]:
			print('\nPolynomial expansion with '+str(algorithm['options']['poly'][1])+' base')
			poly = PolynomialFeatures(algorithm['options']['poly'][1])
			train_reptweets = poly.fit_transform(train_reptweets)
			poly = PolynomialFeatures(algorithm['options']['poly'][1])
			test_reptweets = poly.fit_transform(test_reptweets)

	elif algorithm['options']['feature_extraction'] == 'TFIDF':
		print('\nFeature extraction using TF-IDF\n')
		train_reptweets, test_reptweets = load_vectorizer(tweets, test_tweets)
	elif (algorithm['options']['feature_extraction'] == 'DOC2VEC'):
		print('\tUsing doc2vec')
		train_reptweets, test_reptweets = doc2vec(tweets, test_tweets)



if 'model_selection' in algorithm['options']:
	if algorithm['options']['model_selection']:
		# Parameters Initialization
		if algorithm['options']['ml_algorithm'] == 'SVM':
			print('SVM params init')
			listOLists = [['hinge','squared_hinge'],[0.1,0.5,0.8,1,2]]
			clf = svm.LinearSVC(max_iter=10000)
		elif algorithm['options']['ml_algorithm'] == 'LR':
			print('Logistic params init')
			listOLists = [[1e4, 1e5]]
		elif algorithm['options']['ml_algorithm'] == 'NN':
			print('NN params init')
			listOLists = [[1e-3, 1e-4, 1e-5],\
							['constant'], #,'invscaling', 'adaptive'],\
							['lbfgs', 'adam'],\
							[1,2],\
							[16,32]]#,64,128]]
		c = itertools.product(*listOLists)
		print('combinations: ',c)

		max_tuple = []
		max_avg_score = 0
		for tuple_ in c:
			print('tuple: ',tuple_)
			# Apply Machine Learning Algorithm with cross validation to perform model selection
			if algorithm['options']['ml_algorithm'] == 'SVM':
				print('Initializing SVM')
				clf = svm.LinearSVC(max_iter=10000,intercept_scaling=tuple_[1],loss=tuple_[0])
			elif algorithm['options']['ml_algorithm'] == 'LR':
				print('Initializing Logistic Regression')
				clf = linear_model.LogisticRegression(C=tuple_[0],n_jobs=-1,max_iter=10000)
			elif algorithm['options']['ml_algorithm'] == 'NN':
				print('Initializing Neural Network')
				clf = MLPClassifier(solver=tuple_[2],\
									activation='logistic', \
									hidden_layer_sizes=(tuple_[4], tuple_[3]), \
									random_state=4, \
									verbose=False,\
									alpha=tuple_[0],\
									learning_rate = tuple_[1])

		
			print('Cross-validating results')
			avg_test_accuracy, cv = cross_validation(clf, 
													tweets.shape[0],
													train_reptweets,
													tweets['sentiment'], 
													n_folds=algorithm['options']['cv'][1])
			print('Avg CV score: ',avg_test_accuracy)

			if avg_test_accuracy > max_avg_score:
				max_tuple = tuple_
				max_avg_score = avg_test_accuracy

		# best score of selected machine learning algorithm
		print('max_avg_score', max_avg_score)
		# best parameters for selected machine learning algorithm
		print('max_tuple', max_tuple)


	## best parameters (hardcoded)
	else:
		# Initialize parameters of selected Machine Learning Algorithm
		if algorithm['options']['ml_algorithm'] == 'RF':
			print('\nInitializing Random Forest')
			clf = RandomForestClassifier(n_estimators=algorithm['options']['params']['n_estimators'],\
										 max_depth=algorithm['options']['params']['max_depth'],\
										 n_jobs=-1,random_state=4)
		if algorithm['options']['ml_algorithm'] == 'NB':
			print('\nInitializing Naive Bayes')
			clf = MultinomialNB(alpha=algorithm['params']['alpha'],\
								fit_prior=algorithm['params']['fit_prior'],\
								class_prior=algorithm['params']['class_prior'])
		elif algorithm['options']['ml_algorithm'] == 'SVM':
			print('\nInitializing SVM')
			clf = svm.LinearSVC(max_iter=algorithm['params']['max_iter'],\
			 					intercept_scaling=algorithm['params']['intercept_scaling'],\
			 					loss=algorithm['params']['loss'])
		elif algorithm['options']['ml_algorithm'] == 'LR':
			print('\nInitializing Logistic Regression')
			clf = linear_model.LogisticRegression(C=algorithm['params']['C'],\
								max_iter=algorithm['params']['max_iter'],n_jobs=-1)
		elif algorithm['options']['ml_algorithm'] == 'NN':
			print('\nInitializing Neural Network')
			clf = MLPClassifier(solver=algorithm['params']['solver'],\
								activation=algorithm['params']['activation'], \
								hidden_layer_sizes=(algorithm['params']['k'],algorithm['params']['hidden_layers']),\
								random_state=4, verbose=False,\
								max_iter=algorithm['params']['max_iter'], tol=algorithm['params']['tol'])
if algorithm['options']['ml_algorithm'] == 'CNN':
	if algorithm['params']['train']:
		labels = np.zeros((tweets.shape[0], 2))
		labels[pos_tweets.shape[0]:, 0] = 1.0
		labels[:pos_tweets.shape[0], 1] = 1.0
		if algorithm['params']['train_from'] == 'from_scratch':
			path = trainCNN(tweets, labels, algorithm['params'])
		elif algorithm['params']['train_from'] == 'from_checkpoint':
			path = trainCNN_fromcheckpoint(tweets, labels, algorithm['params'])
	else:
		path = algorithm['params']['checkpoint_dir']
		algorithm['params']['save_from_file'] = True
	pred = evalCNN(test_tweets, path, algorithm['params'])
if algorithm['options']['ml_algorithm'] == 'FT':
	print('\Running FastText')
	pred = fast_text(tweets, test_tweets)
	pred = np.array(pred)

# In case cv is enabled, Cross Validation is performed
if 'cv' in algorithm['options']:
	if algorithm['options']['cv'][0]:
		print('Cross Validation...')
		avg_test_accuracy, cv = cross_validation(clf, 
												tweets.shape[0],
												train_reptweets,
												tweets['sentiment'], 
												n_folds=algorithm['options']['cv'][1])
		print('Avg CV score: ',avg_test_accuracy)

# Train selected model
if algorithm['options']['ml_algorithm'] not in ['FT','CNN']:
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
