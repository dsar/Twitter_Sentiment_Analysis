import numpy as np
import csv
import os

from options import *

def create_csv_submission(y_pred):
    """
    DESCRIPTION: 
            Creates the final submission file to be uploaded on Kaggle platform
    INPUT: 
            y_pred: List of sentiment predictions. Contains 1 and -1 values
    """
    with open(PRED_SUBMISSION_FILE, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        r1 = 1
        for r2 in y_pred:
            writer.writerow({'Id':int(r1),'Prediction':r2})
            r1 += 1

def clear_cache():
	"""
	DESCRIPTION: 
	    Clears the selected cached files from options.py file
	"""
	print('clearing cache files')
	if algorithm['options']['clear_params']['preproc']:
		if os.system('rm '+ PREPROC_DATA_PATH+'*') == 0:
			print('clear preproc DONE')
	if algorithm['options']['clear_params']['tfidf']:
		if os.system('rm ' + TFIDF_TRAIN_FILE) == 0:
			print('clear tfidf DONE')
	if algorithm['options']['clear_params']['pred']:
		if os.system('rm ' + PRED_SUBMISSION_FILE) == 0:
			print('clear pred DONE')
	if algorithm['options']['clear_params']['d2v']:
		if os.system('rm ' + DOC2VEC_MODEL_PATH) == 0:
			print('clear d2v DONE')
	if algorithm['options']['clear_params']['merged']:
		if os.system('rm ' + MERGED_EMBEDDINGS_FILE) == 0:
			print('clear merged DONE')
	if algorithm['options']['clear_params']['baseline_embeddings']:
		if os.system('rm ' + MY_EMBEDDINGS_TXT_FILE) == 0:
			print('clear my txt embeddings DONE')
	if algorithm['options']['clear_params']['my_glove_python_embeddings']:
		if os.system('rm ' + MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE) == 0:
			print('clear my glove python embeddings DONE')
	if algorithm['options']['clear_params']['init_files']:
		if os.system('rm ' + VOCAB_CUT_FILE + ' ' +VOCAB_FILE + ' ' + COOC_FILE+ ' ' + GLOVE_DATA_PATH+'vocab.txt') == 0:
			print('clear my init_files DONE')
	print('clear cache operation... DONE\n')

def read_file(filename):
    """
    DESCRIPTION: 
            Reads a file and returns it as a list
    INPUT: 
            filename: Name of the file to be read
    """
    data = []
    with open(filename, "r") as ins:
        for line in ins:
            data.append(line)
    return data
