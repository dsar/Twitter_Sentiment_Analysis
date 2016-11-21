DATA_PATH = '../data/'
POS_TWEETS_FILE = 'train_pos_small.txt'
NEG_TWEETS_FILE = 'train_neg_small.txt'
TEST_TWEETS_FILE = 'test_data.txt'
PRED_SUBMISSION_FILE = 'pred_submission.csv'

preprocessing_params = {
    'fduplicates': True,
    'frepeated_chars': True,
    'fpunctuation': True,
    'fuser': True,
    'furl': True,
    'fhashtag': True,
    'fdigits': True,
    'fsmall_words': True,
    'save': False
}

vectorizer_params = {
    'min_df' : 5,
    'max_df' : 0.8,
    'sublinear_tf' : True,
    'use_idf' : True,
    'number_of_stopwords' : 10, # None or Int
    'tokenizer' : True, # None or anything else (e.g. True)
    'ngram_range' : (1,1), # (1,2) for bigrams
    'max_features' : None # int
}

kfold = {
    'naive_bayes' : 5,
    'random_forest' : 5,
    'svm' : 5
}

split_params = {
    'test_size' : 0.10,
    'random_state': 4
}

def print_dict_settings(dict_, msg='settings\n'):
    print(msg)
    for key, value in dict_.items():
        print(key,':\t',value)
    print('-\n')