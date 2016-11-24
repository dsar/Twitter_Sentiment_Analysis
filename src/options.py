DATA_PATH = '../data/'
POS_TWEETS_FILE = 'train_pos_full.txt'
NEG_TWEETS_FILE = 'train_neg_full.txt'
TEST_TWEETS_FILE = 'test_data.txt'
PRED_SUBMISSION_FILE = 'pred_submission.csv'

options = {
    'warnings' : False
}

WE_params = {
    'init' : True,
    'we_features' : 50
}

preprocessing_params = {
    'preprocess' : True,
    'fduplicates': True,
    'frepeated_chars': True,
    'fpunctuation': True,
    'fuser': True,
    'furl': True,
    'fhashtag': True,
    'fdigits': True,
    'fsmall_words': False,
    'save': True
}

vectorizer_params = {
    'min_df' : 5,
    'max_df' : 0.8,
    'sublinear_tf' : True,
    'use_idf' : True,
    'number_of_stopwords' : 153, # None or Int (max=153)
    'tokenizer' : True, # None or anything else (e.g. True) for lemmatization
    'ngram_range' : (1,1), # (1,2) for bigrams
    'max_features' : 5000 # None or Int
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
