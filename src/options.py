DATA_PATH = '../data/'
GLOVE_DATA_PATH = '../data/glove_data/'
POS_TWEETS_FILE = 'train_pos_small.txt'
NEG_TWEETS_FILE = 'train_neg_small.txt'
TEST_TWEETS_FILE = 'test_data.txt'
PRED_SUBMISSION_FILE = 'pred_submission.csv'
#remove data/train_preproc_set.csv before starting with a new dataset
TRAIN_PREPROC_CACHING_PATH = 'train_preproc_set.csv'
TEST_PREPROC_CACHING_PATH = 'test_preproc_set.csv'
EMBEDDINGS_FILE_25 = 'glove.twitter.27B.25d.txt'
EMBEDDINGS_FILE_200 = 'glove.twitter.27B.200d.txt'

options = {
    'preprocess' : True,
    'init' : False,
    'ml_algorithm' : 'NN', # {SVM, LR, RF, NN} later will be change to a set
    'feature_extraction' : 'TFIDF', # {TFIDF,WE} later will change to set
    'cv' : (True,5),
    'scale': True,
    'warnings' : False,
    'PCA': (False, 25),
    'poly': (False,2)
}

WE_params = {
    'we_features' : 25,
    'epochs' : 10
}

preprocessing_params = {
    'frepeated_chars': True,
    'fexpand_not': True,
    'fhashtag': True,
    'fdigits': True,
    'fsmall_words': True,
    'fstopwords' : True,
    'fduplicates': False,
    'fpunctuation': False,
    'fuser': False,
    'furl': False,
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
