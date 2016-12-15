import numpy as np
import os
from glove import Corpus, Glove

from options import *

## Functions for word embeddings

def load_glove_embeddings_from_txt_file(filename):
    print('Loading', filename ,'embeddings file')

    if not os.path.exists(filename):
        print(filename,'embeddings not found')
        return None
    print('Creating', filename ,'embeddings dictionary')
    words = {} #key= word, value=embeddings
    with open(filename, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    print('DONE')
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
        words = load_glove_embeddings_from_txt_file(MERGED_EMBEDDINGS_FILE)
        if words != None:
            return words
        print('Merged embeddings file not found')
        my_words = build_glove_embeddings(build_python_glove_representation(tweets['tweet']))
        # glove_words = load_glove_embeddings_from_txt_file(PRETRAINED_EMBEDDINGS_FILE)
        words = build_merge_embeddings()
    return words

def build_merge_embeddings():
	print('Build merged Embeddings')	
	# for k,v in my_words.items():
	# 	if k not in set(glove_words.keys()):
	# 		glove_words[k] = v
	# os.system('sort' + PRETRAINED_EMBEDDINGS_FILE + ' -o ' + PRETRAINED_EMBEDDINGS_FILE)
	# os.system('sort' + MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE + ' -o ' + MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE)
	os.system('join -i -a1 -a2 ' +PRETRAINED_EMBEDDINGS_FILE + ' ' + MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE +' 2>/dev/null | cut -d \' \' -f1-'+str(WE_params['we_features'])+" > "+ MERGED_EMBEDDINGS_FILE)
	# store_embeddings_to_txt_file(glove_words, MERGED_EMBEDDINGS_FILE)
	glove_words = load_glove_embeddings_from_txt_file(MERGED_EMBEDDINGS_FILE)
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
	words = load_glove_embeddings_from_txt_file(MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE)
	if words != None:
		return words
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
