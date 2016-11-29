DATA_PATH = '../data/'
#remove data/train_preproc_set.csv before starting with a new dataset
POS_TWEETS_FILE = 'train_pos_small.txt'
NEG_TWEETS_FILE = 'train_neg_small.txt'
TEST_TWEETS_FILE = 'test_data.txt'
PRED_SUBMISSION_FILE = 'pred_submission.csv'
TRAIN_PREPROC_CACHING_PATH = 'train_preproc_set.csv'
TEST_PREPROC_CACHING_PATH = 'test_preproc_set.csv'

options = {
    'preprocess' : True,
    'init' : True,
    'ml_algorithm' : 'LR', # {SVM, LR, RF} later will be change to a set
    'cv' : (True,5),
    'scale': True,
    'warnings' : False
}

WE_params = {
    'we_features' : 20,
    'epochs' : 10
}

preprocessing_params = {
    'fduplicates': False,
    'frepeated_chars': True,
    'fpunctuation': False,
    'fuser': True,
    'furl': True,
    'fhashtag': True,
    'fdigits': True,
    'fsmall_words': False,
    'fstopwords' : (False,100),
    'save': False
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

split_params = {
    'test_size' : 0.10,
    'random_state': 4
}


def print_dict_settings(dict_, msg='settings\n'):
    print(msg)
    for key, value in dict_.items():
        print(key,':\t',value)
    print('-\n')
