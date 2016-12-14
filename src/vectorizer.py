from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from options import vectorizer_params, print_dict_settings
from preprocessing import *

def init_tfidf_vectorizer():
  print_dict_settings(vectorizer_params, msg='tf-idf Vectorizer settings\n')
  
  if vectorizer_params['number_of_stopwords'] != None:
      vectorizer_params['number_of_stopwords'] = find_stopwords(number_of_stopwords=vectorizer_params['number_of_stopwords'])
  if vectorizer_params['tokenizer'] != None:
      vectorizer_params['tokenizer'] = LemmaTokenizer()

  print('stopwords:\n', vectorizer_params['number_of_stopwords'])

  return TfidfVectorizer(
     min_df = vectorizer_params['min_df'], 
     max_df = vectorizer_params['max_df'], 
     sublinear_tf = vectorizer_params['sublinear_tf'], 
     use_idf = vectorizer_params['use_idf'],
     stop_words = vectorizer_params['number_of_stopwords'], 
     tokenizer = vectorizer_params['tokenizer'],
     ngram_range = vectorizer_params['ngram_range'],
     max_features = vectorizer_params['max_features']
  )

def load_vectorizer(tweets, test_tweets):
  import os.path
  if(os.path.exists('../data/train_reptweets.pkl') and os.path.exists('../data/test_reptweets.pkl')):
    f = open(DATA_PATH+'train_reptweets.pkl','rb')
    train_reptweets = pickle.load(f)
    f = open(DATA_PATH+'test_reptweets.pkl','rb')
    test_reptweets = pickle.load(f)
  else:
    tfidf = init_tfidf_vectorizer()
    train_reptweets = tfidf.fit_transform(tweets['tweet'])
    if options['cache_tfidf']:
      f = open(DATA_PATH+'tfidf_train_reptweets.pkl','wb')
      pickle.dump(train_reptweets, f)
    test_reptweets = tfidf.fit_transform(test_tweets['tweet'])
    if options['cache_tfidf']:
      f = open(DATA_PATH+'tfidf_test_reptweets.pkl','wb')
      pickle.dump(test_reptweets, f)
  return train_reptweets, test_reptweets

def init_count_vectorizer():
  pass
