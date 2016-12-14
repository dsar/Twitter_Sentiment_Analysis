DATA_PATH = '../data/'
PREPROC_DATA_PATH = DATA_PATH+'preproc/'
GLOVE_DATA_PATH = DATA_PATH+'glove_data/'
DOC2VEC_PATH = DATA_PATH + 'doc2vec/'
W2V_DATA_PATH = DATA_PATH + 'word2vec/'

POS_TWEETS_FILE = DATA_PATH+'train_pos.txt'
NEG_TWEETS_FILE = DATA_PATH+'train_neg.txt'
TEST_TWEETS_FILE = DATA_PATH+'test_data.txt'
PRED_SUBMISSION_FILE = DATA_PATH+'pred_submission.csv'
#remove TRAIN_PREPROC_CACHING_PATH before starting with a new dataset
TRAIN_PREPROC_CACHING_PATH = PREPROC_DATA_PATH+'preproc_train.csv'
TEST_PREPROC_CACHING_PATH = PREPROC_DATA_PATH+'preproc_test.csv'
EMBEDDINGS_FILE_25 = GLOVE_DATA_PATH+'glove.twitter.27B.25d.txt'
EMBEDDINGS_FILE_200 = GLOVE_DATA_PATH+'glove.twitter.27B.200d.txt'
MY_EMBEDDINGS_FILE = GLOVE_DATA_PATH+'embeddings.npy'
MERGED_EMBEDDINGS_FILE = GLOVE_DATA_PATH+'merged_embeddings.txt'

DOC2VEC_MODEL_PATH = DOC2VEC_PATH+'paragraph_vector.d2v'

#Sentiment Lexicon
POSITIVE_WORDS=DATA_PATH+'positive-words.txt'
NEGATIVE_WORDS=DATA_PATH+'negative-words.txt'

options = {
    'preprocess' : (True,'save'), #({True,False},{'save', None})
    'init' : 'mytrain', # {'mytrain', 'pretrained', 'merge'}
    'feature_extraction' : 'WE', # {TFIDF,WE} later will change to set
    'we_method' : 'we_mean', # {we_mean, we_tfidf, dm_doc2vec, dbow_doc2vec}
    'ml_algorithm' : 'NN', # {SVM, LR, RF, NN}
    'cv' : (True,5),
    'scale': False,
    'warnings' : False,
    'PCA': (False, 100),
    'poly': (False,2),
    'cache_tfidf': False,
    'model_selection': False,
    'clear' : True
}

WE_params = {
    'we_features' : 200,
    'epochs' : 10
}

clear_params = {
    'preproc' : True,
    'tfidf' : True,
    'pred' : True,
    'd2v' : True,
    'merged': True,
    'my_embeddings' : True
}

NN = {
    'hidden_layers' : 1,
    'k' : 16,
    'solver' : 'adam',
    'activation' : 'logistic',
    'alpha' : 0.0001,
    'learning_rate': 'constant',
    'max_iter': 10000
}

SVM = {
    'loss' : 'squared_hinge',
    'intercept_scaling': 1,
    'max_iter' : 10000
}

LR = {
    'C' : 1e5,
    'max_iter': 10000
}

preprocessing_params = {
    'frepeated_chars': True,
    'fexpand_not': True,
    'transform_emojis': True,
    'fhashtag': True,
    'fdigits': True,
    'sentiment_words': True,

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
    'ngram_range' : (1,1), # (1,2) for bigrams, (1,3) for trigrams and so on
    'max_features' : None # None or Int
}


def print_dict_settings(dict_, msg='settings\n'):
    print(msg)
    for key, value in dict_.items():
        print(key,':\t',value)
    print('-\n')
