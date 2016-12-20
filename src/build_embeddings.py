import numpy as np
import os
from glove import Corpus, Glove

from options import *

## Functions for word embeddings

def load_glove_embeddings_from_txt_file(filename):
    """
    DESCRIPTION: 
            Loads a word embedding file and returns a python dictionary of the form
            (word, [vector of embeddings]) in memory
    INPUT: 
            filename: name of the word embedding file to be loaded
    OUTPUT: 
            words: python dictionary of the form (word, [vector of embeddings])
    """
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
    """
    DESCRIPTION: 
            Loads a word embedding dictionary depending on the selected option in options.py file.
            The options are the following:
                {baseline, glove_python, pretrained, merge}
            baseline: The baseline option builds word embeddings from the training data
                      by the given code on the project's description. This implementation is
                      poor and slow and cannot give us very good results.
            glove_python: This method applies the SGD glove algorithm (proposed by Stanford)
                          and performs really fast and returns a very strong word embedding matrix
                          that is capable of giving us very high accurancy
            pretrained: This method just loads the pretrained embeddings from Stanford
            merge: The last method loads the pretrained word embeddings from Stanford and builds
                    also the word embeddings matrix based on our training dataset by using the 
                    glove_python method. Then all the missing words from the pretrained word
                    embeddings are filled by the glove_python word embeddings.
    INPUT: 
            tweets: tweets Dataframe that contains the tweets from the training dataset
    OUTPUT: 
            words: python dictionary of the form (word, [vector of embeddings])
    """
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
        if os.path.exists(PRETRAINED_EMBEDDINGS_FILE):
            print(PRETRAINED_EMBEDDINGS_FILE,'exists!')
        else:
            print(PRETRAINED_EMBEDDINGS_FILE,'does not exist. Please download Stanford pretrained embeddings file.')
            exit()
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
	"""
	DESCRIPTION: 
	    Loads the pretrained word embeddings from Stanford and builds
	    also the word embeddings matrix based on our training dataset by using the 
	    glove_python method. Then all the missing words from the pretrained word
	    embeddings are filled by the glove_python word embeddings.
	OUTPUT: 
	    glove_words: merged python dictionary of the form (word, [vector of embeddings])
	"""
	print('Build merged Embeddings')	
	os.system('join -i -a1 -a2 ' +PRETRAINED_EMBEDDINGS_FILE + ' ' + MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE +' 2>/dev/null | cut -d \' \' -f1-'+str(algorithm['options']['WE']['we_features'])+" > "+ MERGED_EMBEDDINGS_FILE)
	glove_words = load_glove_embeddings_from_txt_file(MERGED_EMBEDDINGS_FILE)
	return glove_words

def call_init():
	"""
	DESCRIPTION: 
	    Builds the baseline word embeddings.
	    Calls all the required files given in the project's description
	    in order to build the baseline word embeddings.
	OUTPUT: 
	    words: python dictionary of the form (word, [vector of embeddings])
	"""
	words = load_glove_embeddings_from_txt_file(MY_EMBEDDINGS_TXT_FILE)
	if words != None:
		return words
	print('start init.sh')
	os.system('bash init.sh ' + POS_TWEETS_FILE + ' ' + NEG_TWEETS_FILE)
	print('baseline embeddings created')
	return load_glove_embeddings_from_txt_file(MY_EMBEDDINGS_TXT_FILE)

def build_python_glove_representation(tweets):
    """
    DESCRIPTION: 
            Converts initial tweet representation (pandas Dataframe) 
            on the required representation for glove_python algorithm. 
    OUTPUT: 
            A list of lists that contains all the training tweets
    """
    return tweets.apply(lambda tweet: tweet.split()).tolist()

def build_glove_embeddings(corpus):
    """
    DESCRIPTION: 
             Applies the Glove python SGD algorithm given by glove_python library and build the
             word embeddings from our training set.
    INPUT:
            corpus: a list of lists where each sub-list represent a tweet. The outer list represents
                    the whole training dataset.
    OUTPUT: 
            words: python dictionary of the form (word, [vector of embeddings])
    """
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
	"""
	DESCRIPTION: 
	     Stores a python dictionary of the form (word, [vector of embeddings]) (which represents
	     the word embeddings matrix of our model) to a txt file. 
	INPUT:
	    dict: python dictionary of the form (word, [vector of embeddings])
	    filename: name of the file to write the word embeddings dictionary
	"""
	with open(filename, "w") as f:
		for k, v in dict.items():
			line = k + str(v) + '\n'
			f.write(str(k+' '))
			for i in v:
				f.write("%s " % i)
			f.write('\n')
