import numpy as np
import csv
import os
from glove import Corpus, Glove

from options import *

def create_csv_submission(y_pred):
    with open(PRED_SUBMISSION_FILE, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        r1 = 1
        for r2 in y_pred:
            writer.writerow({'Id':int(r1),'Prediction':r2})
            r1 += 1

def clear_cache(clear):
	print('clearing cache files')
	if clear['preproc']:
		os.system('rm '+ PREPROC_DATA_PATH+'*')
		print('\nclear preproc DONE\n')
	if clear['tfidf']:
		os.system('rm ' + DATA_PATH+'tfidf_*_reptweets.pkl')
		print('\nclear tfidf DONE\n')
	if clear['pred']:
		os.system('rm ' + PRED_SUBMISSION_FILE)
		print('\nclear pred DONE\n')
	if clear['d2v']:
		os.system('rm ' + DOC2VEC_MODEL_PATH)
		print('\nclear d2v DONE\n')
	if clear['merged']:
		os.system('rm ' + MERGED_EMBEDDINGS_FILE)
		print('\nclear merged DONE\n')
	if clear['my_embeddings']:
		os.system('rm ' + MY_EMBEDDINGS_FILE)
		print('\nclear my embeddings DONE\n')
	print('\nclear cache DONE\n')

def read_file(filename):
    data = []
    with open(filename, "r") as ins:
        for line in ins:
            data.append(line)
    return data


## Functions for word embeddings

def load_my_embeddings():
    words = {} #key= word, value=embeddings
    we = np.load(MY_EMBEDDINGS_FILE)
    print('we shape', we.shape)
    vocab_file = open(DATA_PATH+'vocab_cut.txt', "r")
    for i, line in enumerate(vocab_file):
        words[line.rstrip()] = we[i]
    return words

def load_glove_embeddings():
    words = {} #key= word, value=embeddings
    with open(EMBEDDINGS_FILE_200, "r") as f:
        for line in f:
            tokens = line.strip().split()
            words[tokens[0]] = np.array([float(x) for x in tokens[1:]])
    return words

def get_embeddings_dictionary(tweets=None):
    if options['init'] == 'mytrain':
        # call_init()
        # words = load_my_embeddings()
        words = build_glove_embeddings(build_python_glove_representation(tweets['tweet']))
    elif options['init'] == 'pretrained':
        words = load_glove_embeddings()
    elif options['init'] == 'merge':
        call_init()
        my_words = load_my_embeddings()
        glove_words = load_glove_embeddings()
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
		with open(MERGED_EMBEDDINGS_FILE, "w") as f:
			for k, v in glove_words.items():
				line = k + str(v) + '\n'
				f.write(str(k+' '))
				for i in v:
					f.write("%s " % i)
				f.write('\n')
	print(glove_words)
	return glove_words

def call_init():
	if not os.path.exists(MY_EMBEDDINGS_FILE):
		print('start init.sh')
		os.system('bash init.sh ' + POS_TWEETS_FILE + ' ' + NEG_TWEETS_FILE)
	else:
		print('my embeddings found')

def build_python_glove_representation(tweets):
    """
    tweets Series (column)
    """
    return tweets.apply(lambda tweet: tweet.split()).tolist()

def build_glove_embeddings(corpus):
	model  = Corpus()
	model.fit(corpus, window = 5)

	glove = Glove(no_components=200, learning_rate=0.1)
	print('fitting Glove')
	glove.fit(model.matrix, epochs=10)
	glove.add_dictionary(model.dictionary)

	words = {}
	for w, id_ in glove.dictionary.items():
		words[w] = np.array(glove.word_vectors[id_])

	return words
