import numpy as np
import os
from glove import Corpus, Glove

from options import *

## Functions for word embeddings

def load_glove_embeddings_from_txt_file(filename):
    print('Loading embeddings from txt')
    if not os.path.exists(filename):
        print('embeddings not found')
        return None
    print('Creating embeddings dictionary')
    words = {} #key= word, value=embeddings
    with open(filename, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    return words

def get_embeddings_dictionary(tweets=None):
    if options['build_we_method'] == 'baseline':
        words = call_init()
    elif options['build_we_method'] == 'glove_python':
        words = build_glove_embeddings(build_python_glove_representation(tweets['tweet']))
    elif options['build_we_method'] == 'pretrained':
        words = load_glove_embeddings_from_txt_file(PRETRAINED_EMBEDDINGS_FILE)
    elif options['build_we_method'] == 'merge':
        # my_words = call_init()
        my_words = build_glove_embeddings(build_python_glove_representation(tweets['tweet']))
        glove_words = load_glove_embeddings_from_txt_file(PRETRAINED_EMBEDDINGS_FILE)
        words = merge_embeddings(glove_words, my_words)
    return words

def merge_embeddings(glove_words, my_words):
	if os.path.exists(MERGED_EMBEDDINGS_FILE):
		print('Load merged Embeddings')
		glove_words = {}
		with open(MERGED_EMBEDDINGS_FILE, "r") as f:
			for line in f:
				tokens = line.strip().split()
				glove_words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
	else:
		print('Build merged Embeddings')	
		for k,v in my_words.items():
			if k not in set(glove_words.keys()):
				glove_words[k] = v
		store_embeddings_to_txt_file(glove_words, MERGED_EMBEDDINGS_FILE)
	print(glove_words)
	return glove_words

def call_init():
	words = load_glove_embeddings_from_txt_file(MY_EMBEDDINGS_TXT_FILE)
	if words != None:
		return words
	print('start init.sh')
	os.system('bash init.sh ' + POS_TWEETS_FILE + ' ' + NEG_TWEETS_FILE)
	print('baseline embeddings created')
	return load_glove_embeddings_from_txt_file(MY_EMBEDDINGS_TXT_FILE)

def build_python_glove_representation(tweets):
    """
    tweets Series (column)
    """
    return tweets.apply(lambda tweet: tweet.split()).tolist()

def build_glove_embeddings(corpus):
	print('Loading embeddings')
	words = load_glove_embeddings_from_txt_file(MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE)
	if words != None:
		print('DONE\n')
		return words
	print('Embeddings file not found')
	model  = Corpus()
	model.fit(corpus, window = WE_params['window_size'])

	glove = Glove(no_components=WE_params['we_features'], learning_rate=WE_params['learning_rate'])
	print('\nFitting Glove Python Embeddings')
	glove.fit(model.matrix, epochs=WE_params['epochs'])
	glove.add_dictionary(model.dictionary)

	words = {}
	for w, id_ in glove.dictionary.items():
		words[w] = np.array(glove.word_vectors[id_])

	store_embeddings_to_txt_file(words, MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE)
	return words

def store_embeddings_to_txt_file(dict, filename):
	with open(filename, "w") as f:
		for k, v in dict.items():
			line = k + str(v) + '\n'
			f.write(str(k+' '))
			for i in v:
				f.write("%s " % i)
			f.write('\n')
