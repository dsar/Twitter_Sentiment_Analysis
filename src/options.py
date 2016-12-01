DATA_PATH = '../data/'
#remove data/train_preproc_set.csv before starting with a new dataset
POS_TWEETS_FILE = 'train_pos_full.txt'
NEG_TWEETS_FILE = 'train_neg_full.txt'
TEST_TWEETS_FILE = 'test_data.txt'
PRED_SUBMISSION_FILE = 'pred_submission.csv'
TRAIN_PREPROC_CACHING_PATH = 'train_preproc_set.csv'
TEST_PREPROC_CACHING_PATH = 'test_preproc_set.csv'
EMBEDDINGS_FILE = 'embeddings_200d_full.npy'

options = {
    'preprocess' : False,
    'init' : False,
    'ml_algorithm' : 'LR', # {SVM, LR, RF} later will be change to a set
    'feature_extraction' : 'WE', #later will change to set
    'cv' : (False,5),
    'scale': False,
    'warnings' : False,
    'PCA': (False, 50)
}

WE_params = {
    'we_features' : 200,
    'epochs' : 10
}

preprocessing_params = {
    'fduplicates': False,
    'frepeated_chars': True,
    'fexpand_not': True,
    'fpunctuation': False,
    'fuser': False,
    'furl': False,
    'fhashtag': True,
    'fdigits': False,
    'fsmall_words': False,
    'fstopwords' : False,
    'save': True
}

vectorizer_params = {
    'min_df' : 5,
    'max_df' : 0.8,
    'sublinear_tf' : True,
    'use_idf' : True,
    'number_of_stopwords' : None, # None or Int
    'tokenizer' : True, # None or anything else (e.g. True) for lemmatization
    'ngram_range' : (1,1), # (1,2) for bigrams
    'max_features' : None # None or Int
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
