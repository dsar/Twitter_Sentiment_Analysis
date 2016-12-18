#Paths
DATA_PATH = '../data/'
DATASETS_PATH = DATA_PATH + 'datasets/'
SUBMISSIONS_PATH = DATA_PATH + 'submissions/'
METADATA_PATH = DATA_PATH + 'meta/'
PREPROC_DATA_PATH = DATA_PATH+'preproc/'
GLOVE_DATA_PATH = DATA_PATH+'glove/'
DOC2VEC_PATH = DATA_PATH + 'doc2vec/'
W2V_DATA_PATH = DATA_PATH + 'word2vec/'
FASTTEXT_DATA_PATH = DATA_PATH + 'fasttext/'
TFIDF_DATA_PATH = DATA_PATH + 'tfidf/'

POS_TWEETS_FILE = DATASETS_PATH + 'train_pos_full.txt'
NEG_TWEETS_FILE = DATASETS_PATH + 'train_neg_full.txt'
TEST_TWEETS_FILE = DATASETS_PATH + 'test_data.txt'
PRED_SUBMISSION_FILE = SUBMISSIONS_PATH + 'pred_submission.csv'
#remove TRAIN_PREPROC_CACHING_PATH before starting with a new dataset
TRAIN_PREPROC_CACHING_PATH = PREPROC_DATA_PATH + 'preproc_train.csv'
TEST_PREPROC_CACHING_PATH = PREPROC_DATA_PATH + 'preproc_test.csv'
PRETRAINED_EMBEDDINGS_FILE = GLOVE_DATA_PATH + 'glove.twitter.27B.200d.txt'
MY_EMBEDDINGS_TXT_FILE = GLOVE_DATA_PATH + 'baseline_embeddings.txt'
MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE = GLOVE_DATA_PATH + 'glove_python_embeddings.txt'
MERGED_EMBEDDINGS_FILE = GLOVE_DATA_PATH + 'merged_embeddings.txt'
FASTTEXT_TRAIN_FILE = FASTTEXT_DATA_PATH + 'fasttext_train.txt'
FASTTEXT_MODEL = FASTTEXT_DATA_PATH + 'fasttext_model'
TF_SAVE_PATH = DATA_PATH + 'models/'
TFIDF_TRAIN_FILE = TFIDF_DATA_PATH + 'train_reptweets.pkl'
VOCAB_CUT_FILE = GLOVE_DATA_PATH + 'vocab_cut.txt'
VOCAB_FILE = GLOVE_DATA_PATH + 'vocab.pkl'
COOC_FILE = GLOVE_DATA_PATH + 'cooc.pkl'


DOC2VEC_MODEL_PATH = DOC2VEC_PATH+'paragraph_vector.d2v'

#Sentiment Lexicon
POSITIVE_WORDS=METADATA_PATH+'positive-words.txt'
NEGATIVE_WORDS=METADATA_PATH+'negative-words.txt'
WORD_FREQUENCIES = METADATA_PATH + 'words-by-frequency.txt'



# After performing Model Selection we have found the optimal parameters that
# give us the reported scores. The only option that need to be set is the algorithm.
# However, if you want to play with the parameters, go to the corresponding dictionary
# of each algorithm and set it.

algorithm = 'FT' #{SVM, LR, NN, CNN, RF, FT}

SVM = {
    'params' : {
                'loss' : 'squared_hinge',
                'intercept_scaling': 1,
                'max_iter' : 10000
                },
    'options' : {
                    'ml_algorithm' : 'SVM',
                    'preprocess' : (True,'save'), #({True,False},{'save', None})
                    'preprocessing_params' : { # Check the documentation of preprocessing.py functions 
                                                # group1
                                                'frepeated_chars': True,
                                                'fexpand_not': True,
                                                'transform_emojis': True,
                                                'fhashtag': True,
                                                'fdigits': True,
                                                'sentiment_words': True,
                                                # group2
                                                'fsmall_words': False,
                                                'fstopwords' : False,
                                                'fduplicates': False,
                                                'fpunctuation': False,
                                                'fuser': False,
                                                'furl': False
                                              },
                    'feature_extraction' : 'TFIDF', # {TFIDF,WE,DOC2VEC} later will change to set
                    'WE' : {
                              'build_we_method' : 'pretrained', # {'baseline', 'pretrained', 'glove_python', 'merge'}
                              'tweet2vec_method' : 'we_mean', # {we_mean, we_tfidf}
                              'we_features' : 200,
                              'epochs' : 50,
                              'learning_rate' : 0.05,
                              'window_size' : 5
                            },
                    'DOC2VEC' : {
                                  'method' : 'dm_doc2vec', # {dm_doc2vec, dbow_doc2vec}
                                  'we_features' : 200,
                                  'epochs' : 20,
                                  'window_size' : 5
                                },
                    'TFIDF' : {
                                'cache_tfidf': True,

                                'min_df' : 1,
                                'max_df' : 1.0,
                                'sublinear_tf' : True,
                                'use_idf' : True,
                                'number_of_stopwords' : None, # None or Int
                                'tokenizer' : True, # None or anything else (e.g. True) for lemmatization
                                'ngram_range' : (1,1), # (1,2) for bigrams, (1,3) for trigrams and so on
                                'max_features' : None # None or Int
                              },
                    'cv' : (False,5),
                    'scale': False,
                    'warnings' : False,
                    'PCA': (False, 100),
                    'poly': (False,2),
                    'model_selection': False,
                    'clear' : False,
                    'clear_params' : { #In case an update is needed, the corresponding file must be deleted first by enabling the options below.
                                      'preproc' : False,

                                      'tfidf' : False,
                                      'd2v' : False,

                                      'baseline_embeddings' : False,
                                      'my_glove_python_embeddings' : False,
                                      'merged': False,
                                      'init_files' : True,

                                      'pred' : False
                                      }
                }
}

LR = {
    'params' :  {
                'C' : 1e5,
                'max_iter': 10000
                },
        'options' : {
                    'ml_algorithm' : 'LR',
                    'preprocess' : (True,'save'), #({True,False},{'save', None})
                    'preprocessing_params' : { 
                                                # group1
                                                'frepeated_chars': True,
                                                'fexpand_not': True,
                                                'transform_emojis': True,
                                                'fhashtag': True,
                                                'fdigits': True,
                                                'sentiment_words': True,
                                                # group2
                                                'fsmall_words': False,
                                                'fstopwords' : False,
                                                'fduplicates': False,
                                                'fpunctuation': False,
                                                'fuser': False,
                                                'furl': False

                                              },
                    'feature_extraction' : 'WE', # {TFIDF,WE,DOC2VEC} later will change to set
                    'WE' : {
                              'build_we_method' : 'pretrained', # {'baseline', 'pretrained', 'glove_python', 'merge'}
                              'tweet2vec_method' : 'we_mean', # {we_mean, we_tfidf}
                              'we_features' : 200,
                              'epochs' : 50,
                              'learning_rate' : 0.05,
                              'window_size' : 5
                            },
                    'DOC2VEC' : {
                                  'method' : 'dbow_doc2vec', # {dm_doc2vec, dbow_doc2vec}
                                  'we_features' : 200,
                                  'epochs' : 20,
                                  'window_size' : 5
                                },
                    'TFIDF' : {
                                'cache_tfidf': False,

                                'min_df' : 1,
                                'max_df' : 1.0,
                                'sublinear_tf' : True,
                                'use_idf' : True,
                                'number_of_stopwords' : None, # None or Int
                                'tokenizer' : True, # None or anything else (e.g. True) for lemmatization
                                'ngram_range' : (1,1), # (1,2) for bigrams, (1,3) for trigrams and so on
                                'max_features' : None # None or Int
                              },
                    'cv' : (True,5),
                    'scale': False,
                    'warnings' : False,
                    'PCA': (False, 100),
                    'poly': (False,2),
                    'model_selection': False,
                    'clear' : True,
                    'clear_params' : {#In case an update is needed, the corresponding file must be deleted first by enabling the options below.
                                      'preproc' : False,

                                      'tfidf' : True,
                                      'd2v' : True,

                                      'baseline_embeddings' : False,
                                      'my_glove_python_embeddings' : False,
                                      'merged': False,
                                      'init_files' : True,

                                      'pred' : False
                                      }
                }
}

NN = {
    'params' : {  
                  'hidden_layers' : 1,
                  'k' : 16,
                  'solver' : 'lbfgs', #adam is slightly better but lbfgs is much faster
                  'activation' : 'logistic',
                  'alpha' : 1e-5,
                  'learning_rate': 'constant',
                  'max_iter': 10000,
                  'tol' : 1e-4
               },
    'options' : {
                    'ml_algorithm' : 'NN',
                    'preprocess' : (True,'save'), #({True,False},{'save', None})
                    'preprocessing_params' : { #In case an update is needed, the corresponding file must be deleted first by enabling the options below.
                                                # group1
                                                'frepeated_chars': True,
                                                'fexpand_not': True,
                                                'transform_emojis': True,
                                                'fhashtag': True,
                                                'fdigits': True,
                                                'sentiment_words': True,
                                                # group2
                                                'fsmall_words': False,
                                                'fstopwords' : False,
                                                'fduplicates': False,
                                                'fpunctuation': False,
                                                'fuser': False,
                                                'furl': False

                                              },
                    'feature_extraction' : 'WE', # {WE,DOC2VEC} later will change to set
                    'WE' : {
                              'build_we_method' : 'glove_python', # {'baseline', 'pretrained', 'glove_python', 'merge'}
                              'tweet2vec_method' : 'we_mean', # {we_mean, we_tfidf}
                              'we_features' : 200,
                              'epochs' : 50,
                              'learning_rate' : 0.05,
                              'window_size' : 5
                            },
                    'DOC2VEC' : {
                                  'method' : 'dm_doc2vec', # {dm_doc2vec, dbow_doc2vec}
                                  'we_features' : 200,
                                  'epochs' : 20,
                                  'window_size' : 5
                                },
                    'TFIDF' : { # mainly added for we_tfidf method
                                'cache_tfidf': False,

                                'min_df' : 1,
                                'max_df' : 1.0,
                                'sublinear_tf' : True,
                                'use_idf' : True,
                                'number_of_stopwords' : None, # None or Int
                                'tokenizer' : True, # None or anything else (e.g. True) for lemmatization
                                'ngram_range' : (1,1), # (1,2) for bigrams, (1,3) for trigrams and so on
                                'max_features' : None # None or Int
                              },
                    'cv' : (False,5),
                    'scale': False,
                    'warnings' : False,
                    'PCA': (False, 100),
                    'poly': (False,2),
                    'model_selection': False,
                    'clear' : False,
                    'clear_params' : {
                                      'preproc' : False,

                                      'tfidf' : True,
                                      'd2v' : True,

                                      'baseline_embeddings' : False,
                                      'my_glove_python_embeddings' : False,
                                      'merged': False,
                                      'init_files' : True,

                                      'pred' : False
                                      }
                }
}

CNN = {
       'params' : {
                    'embedding_size':200,
                    'n_filters':128,
                    'filter_sizes':[2, 3, 4, 5, 6],
                    'n_layers':1,
                    'n_hidden':128,
                    'n_classes':2,
                    'dropout_prob':0.5,
                    'optimizer':'Adam',  #{Adam, RMSProp}
                    'lambda':5e-4,
                    'lambda_decay_period':500,
                    'lambda_decay_rate':0.97,
                    'moment':0.9,
                    'eval_every':20,
                    'checkpoint_every':1000,
                    'n_checkpoints_to_keep':1,
                    'max_num_words':40,
                    'batch_size':1024,
                    'n_epochs':10,
                    'shuffle_every_epoch':True,
                    'train':True,
                    'train_from':'from_scratch', #{from_checkpoint, from_scratch}
                    'save_from_file':False,
                    'checkpoint_dir': TF_SAVE_PATH + '/1481913588/checkpoints',
                    'n_valid':1000
                  },
        'options' : {
                    'ml_algorithm' : 'CNN',
                    'preprocess' : (True,'save'), #({True,False},{'save', None})
                    'preprocessing_params' : { #In case an update is needed, the corresponding file must be deleted first by enabling the options below.
                                                # group1
                                                'frepeated_chars': True,
                                                'fexpand_not': True,
                                                'transform_emojis': True,
                                                'fhashtag': True,
                                                'fdigits': True,
                                                'sentiment_words': True,
                                                # group2
                                                'fsmall_words': False,
                                                'fstopwords' : False,
                                                'fduplicates': False,
                                                'fpunctuation': False,
                                                'fuser': False,
                                                'furl': False

                                              },
                    'warnings' : False,
                    'clear' : False,
                    'clear_params' : {
                                      'preproc' : False,

                                      'tfidf' : True,
                                      'd2v' : True,

                                      'baseline_embeddings' : False,
                                      'my_glove_python_embeddings' : False,
                                      'merged': False,
                                      'init_files' : True,

                                      'pred' : False
                                      }
                }

}

RF = {
    'params' : {
                  'n_estimators' : 100,
                  'max_depth' : 50
                },
    'options' : {
                    'ml_algorithm' : 'RF',
                    'preprocess' : (True,'save'), #({True,False},{'save', None})
                    'preprocessing_params' : { # Check the documentation of preprocessing.py functions 
                                                # group1
                                                'frepeated_chars': True,
                                                'fexpand_not': True,
                                                'transform_emojis': True,
                                                'fhashtag': True,
                                                'fdigits': True,
                                                'sentiment_words': True,
                                                # group2
                                                'fsmall_words': False,
                                                'fstopwords' : False,
                                                'fduplicates': False,
                                                'fpunctuation': False,
                                                'fuser': False,
                                                'furl': False
                                              },
                    'feature_extraction' : 'TFIDF', # {TFIDF,WE,DOC2VEC} later will change to set
                    'WE' : {
                              'build_we_method' : 'pretrained', # {'baseline', 'pretrained', 'glove_python', 'merge'}
                              'tweet2vec_method' : 'we_mean', # {we_mean, we_tfidf}
                              'we_features' : 200,
                              'epochs' : 50,
                              'learning_rate' : 0.05,
                              'window_size' : 5
                            },
                    'DOC2VEC' : {
                                  'method' : 'dm_doc2vec', # {dm_doc2vec, dbow_doc2vec}
                                  'we_features' : 200,
                                  'epochs' : 20,
                                  'window_size' : 5
                                },
                    'TFIDF' : {
                                'cache_tfidf': True,

                                'min_df' : 1,
                                'max_df' : 1.0,
                                'sublinear_tf' : True,
                                'use_idf' : True,
                                'number_of_stopwords' : None, # None or Int
                                'tokenizer' : True, # None or anything else (e.g. True) for lemmatization
                                'ngram_range' : (1,1), # (1,2) for bigrams, (1,3) for trigrams and so on
                                'max_features' : None # None or Int
                              },
                    'cv' : (False,5),
                    'scale': False,
                    'warnings' : False,
                    'PCA': (False, 100),
                    'poly': (False,2),
                    'model_selection': False,
                    'clear' : False,
                    'clear_params' : { #In case an update is needed, the corresponding file must be deleted first by enabling the options below.
                                      'preproc' : False,

                                      'tfidf' : False,
                                      'd2v' : False,

                                      'baseline_embeddings' : False,
                                      'my_glove_python_embeddings' : False,
                                      'merged': False,
                                      'init_files' : True,

                                      'pred' : False
                                      }
                }
}

FT = {
    'params' : {
                'we_features' : 200,
                'epochs':50, 
                'learning_rate' : 0.05,
                'window_size' : 5
              },
    'options' : { 
                  'ml_algorithm' : 'FT',
                  'preprocess' : (True,'save'), #({True,False},{'save', None})
                  'preprocessing_params' : { #In case an update is needed, the corresponding file must be deleted first by enabling the options below.
                                              # group1
                                              'frepeated_chars': True,
                                              'fexpand_not': True,
                                              'transform_emojis': True,
                                              'fhashtag': True,
                                              'fdigits': True,
                                              'sentiment_words': True,
                                              # group2
                                              'fsmall_words': False,
                                              'fstopwords' : False,
                                              'fduplicates': False,
                                              'fpunctuation': False,
                                              'fuser': False,
                                              'furl': False

                                            },
                  'warnings' : False,
                  'clear' : False,
                  'clear_params' : {
                                    'preproc' : False,

                                    'tfidf' : True,
                                    'd2v' : True,

                                    'baseline_embeddings' : False,
                                    'my_glove_python_embeddings' : False,
                                    'merged': False,
                                    'init_files' : True,

                                    'pred' : False
                                    }
                  }
}

if algorithm == 'SVM':
  algorithm = SVM
elif algorithm == 'LR':
  algorithm = LR
elif algorithm == 'NN':
  algorithm = NN
elif algorithm == 'CNN':
  algorithm = CNN
elif algorithm == 'FT':
  algorithm = FT

def print_dict_settings(dict_, msg='settings\n'):
    print(msg)
    for key, value in dict_.items():
        print('\t',key,':\t',value)
    print('-\n')
