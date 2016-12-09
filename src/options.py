DATA_PATH = '../data/'
PREPROC_DATA_PATH = DATA_PATH+'preproc/'
GLOVE_DATA_PATH = DATA_PATH+'glove_data/'

POS_TWEETS_FILE = DATA_PATH+'train_pos_small.txt'
NEG_TWEETS_FILE = DATA_PATH+'train_neg_small.txt'
TEST_TWEETS_FILE = DATA_PATH+'test_data.txt'
PRED_SUBMISSION_FILE = DATA_PATH+'pred_submission.csv'
#remove data/train_preproc_set.csv before starting with a new dataset
TRAIN_PREPROC_CACHING_PATH = PREPROC_DATA_PATH+'preproc_train.csv'
TEST_PREPROC_CACHING_PATH = PREPROC_DATA_PATH+'preproc_test.csv'
EMBEDDINGS_FILE_25 = GLOVE_DATA_PATH+'glove.twitter.27B.25d.txt'
EMBEDDINGS_FILE_200 = GLOVE_DATA_PATH+'glove.twitter.27B.200d.txt'

DOC2VEC_MODEL_PATH = DATA_PATH+'paragraph_vector.d2v'

options = {
    'preprocess' : (True,'save'), #({True,False},{'save'})
    'init' : False,
    'feature_extraction' : 'WE', # {TFIDF,WE} later will change to set
    'we_method' : 'baseline', # {baseline, doc2vec}
    'ml_algorithm' : 'SVM', # {SVM, LR, RF, NN} later will be change to a set
    'cv' : (True,5),
    'scale': True,
    'warnings' : False,
    'PCA': (False, 25),
    'poly': (False,2),
    'cache_tfidf': False,
    'clear' : True
}

WE_params = {
    'we_features' : 25,
    'epochs' : 10
}

preprocessing_params = {
    'frepeated_chars': True,
    'fexpand_not': True,
    'transform_emojis': True,
    'fhashtag': True,
    'fdigits': True,

    'fsmall_words': False,
    'fstopwords' : False,
    'fduplicates': False,
    'fpunctuation': False,
    'fuser': False,
    'furl': False,
}

vectorizer_params = {
    'min_df' : 1,
    'max_df' : 1.0,
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
