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

def init_count_vectorizer():
  pass