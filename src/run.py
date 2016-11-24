import numpy as np
import pandas as pd
import csv
from preprocessing import preprocessing
from options import *

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def remove_tweet_id(tweet):
    return tweet.split(',', 1)[-1]

pos_tweets = pd.read_table(POS_TWEETS_FILE, names=['tweet','sentiment'], low_memory=False)
pos_tweets['sentiment'] = 'pos'
neg_tweets = pd.read_table(NEG_TWEETS_FILE ,names=['tweet','sentiment'], low_memory=False)
neg_tweets['sentiment'] = 'neg'

tweets = pd.concat([pos_tweets, neg_tweets], axis=0)

print('preprocessing')
tweets = preprocessing(tweets,train=True, params=preprocessing_params)

test_tweets = pd.read_table(TEST_TWEETS_FILE, names=['tweet','sentiment'])
test_tweets['tweet'] = test_tweets.apply(lambda tweet: remove_tweet_id(tweet['tweet']), axis=1)
test_tweets = preprocessing(test_tweets,train=False, params=preprocessing_params)

print('final representation shape', tweets.shape)

we = np.load('embeddings.npy')

vocab = pd.read_table('vocab_cut.txt',header=None, names=['word'])

words = vocab['word'].to_dict()
print('reverse')
words = {v: k for k, v in words.items()}

print('tweets20')
tweets20 = np.zeros((tweets.shape[0],20))
i = 0
for tweet in tweets['tweet']:
    split_tweet = tweet.split()
    vec = np.zeros(20)
    for word in split_tweet:
        try:
            vec += we[words[word]]
        except:
            vec += 0
    #vec /= len(split_tweet)
    tweets20[i] = vec
    i+=1


print('tweets_test')
tweets_test = np.zeros((test_tweets.shape[0],20))
i = 0
for tweet in test_tweets['tweet']:
    split_tweet = tweet.split()
    vec = np.zeros(20)
    for word in split_tweet:
        try:
            vec += we[words[word]]
        except:
            vec += 0
    #vec /= len(split_tweet)
    tweets_test[i] = vec
    i+=1

def change(row):
    if row == 'pos': 
        return 1 
    return -1
    
y = tweets.apply(lambda row: change(row['sentiment']), axis=1)


print('SVM')
#classifier_linear = svm.SVC(kernel='linear')
#classifier_linear.fit(tweets20, y)
#pred = classifier_linear.predict(tweets_test)

print('RF')
forest = RandomForestClassifier(n_estimators=100,max_depth=50,n_jobs=-1,random_state=4)
forest.fit(tweets20, y)

tweets_test = np.nan_to_num(tweets_test)  #!!!!!!! under discussion
pred = forest.predict(tweets_test)

print('pred shape: ',pred.shape)
print('pred values: ',pred)

def create_csv_submission(y_pred, name):
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        r1 = 1
        for r2 in y_pred:
            if r2 == 1:
                writer.writerow({'Id':int(r1),'Prediction':1})
            elif r2 == -1:
                writer.writerow({'Id':int(r1),'Prediction':-1})
            r1+=1

print('writing submission')
create_csv_submission(pred, PRED_SUBMISSION_FILE)
