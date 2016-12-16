import numpy as np
import csv
import os

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
		if os.system('rm '+ PREPROC_DATA_PATH+'*') == 0:
			print('clear preproc DONE')
	if clear['tfidf']:
		if os.system('rm ' + DATA_PATH+'tfidf_*_reptweets.pkl') == 0:
			print('clear tfidf DONE')
	if clear['pred']:
		if os.system('rm ' + PRED_SUBMISSION_FILE) == 0:
			print('clear pred DONE')
	if clear['d2v']:
		if os.system('rm ' + DOC2VEC_MODEL_PATH) == 0:
			print('clear d2v DONE')
	if clear['merged']:
		if os.system('rm ' + MERGED_EMBEDDINGS_FILE) == 0:
			print('clear merged DONE')
	if clear['baseline_embeddings']:
		if os.system('rm ' + MY_EMBEDDINGS_TXT_FILE) == 0:
			print('clear my txt embeddings DONE')
	if clear['my_glove_python_embeddings']:
		if os.system('rm ' + MY_GLOVE_PYTHON_EMBEDDINGS_TXT_FILE) == 0:
			print('clear my glove python embeddings DONE')
	print('clear cache operation... DONE\n')

def read_file(filename):
    data = []
    with open(filename, "r") as ins:
        for line in ins:
            data.append(line)
    return data