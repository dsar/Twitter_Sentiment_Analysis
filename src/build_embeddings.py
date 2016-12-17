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
    print('Constructing dictionary for', filename, 'file')
    words = {} #key= word, value=embeddings
    with open(filename, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    print('DONE')
    return words

def get_embeddings_dictionary(tweets=None):
    if algorithm['options']['ml_algorithm'] == 'CNN':
        return load_glove_embeddings_from_txt_file(PRETRAINED_EMBEDDINGS_FILE)
    if tweets is None:
        print('WARNING(!): Tweets is None.')
    if algorithm['options']['WE']['build_we_method'] == 'baseline':
        words = call_init()
    elif algorithm['options']['WE']['build_we_method'] == 'glove_python':
        new_tweets_representation = build_python_glove_representation(tweets['tweet'])
        words = build_glove_embeddings(new_tweets_representation)
    elif algorithm['options']['WE']['build_we_method'] == 'pretrained':
        words = load_glove_embeddings_from_txt_file(PRETRAINED_EMBEDDINGS_FILE)
    elif algorithm['options']['WE']['build_we_method'] == 'merge':
        # my_words = call_init()
        words = load_glove_embeddings_from_txt_file(MERGED_EMBEDDINGS_FILE)
        if words != None:
            return words
        print('Merged embeddings file not found')
        if not os.path.exists(MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE):
        	build_glove_embeddings(build_python_glove_representation(tweets['tweet']))
        else:
        	print(MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE,'exists!')
        if os.path.exists(PRETRAINED_EMBEDDINGS_FILE):
        	print(PRETRAINED_EMBEDDINGS_FILE,'exists!')
        else:
        	print(PRETRAINED_EMBEDDINGS_FILE,'does not exist. Please download Stanford pretrained embeddings file.')
        	exit()
        # glove_words = load_glove_embeddings_from_txt_file(PRETRAINED_EMBEDDINGS_FILE)
        words = build_merge_embeddings()
    return words

def build_merge_embeddings():
	print('Build merged Embeddings')	
	os.system('join -i -a1 -a2 ' +PRETRAINED_EMBEDDINGS_FILE + ' ' + MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE +' 2>/dev/null | cut -d \' \' -f1-'+str(algorithm['options']['WE']['we_features'])+" > "+ MERGED_EMBEDDINGS_FILE)
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
    model.fit(corpus, window = algorithm['options']['WE']['window_size'])

    glove = Glove(no_components=algorithm['options']['WE']['we_features'], learning_rate=algorithm['options']['WE']['learning_rate'])
    print('\nFitting Glove Python Embeddings')
    glove.fit(model.matrix, epochs=algorithm['options']['WE']['epochs'])
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
