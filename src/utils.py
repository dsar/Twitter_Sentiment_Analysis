import csv
from options import *
import os

def create_csv_submission(y_pred):
    with open(PRED_SUBMISSION_FILE, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        r1 = 1
        for r2 in y_pred:
            writer.writerow({'Id':int(r1),'Prediction':r2})
            r1 += 1

def clear_cache(preproc=True,tfidf=True,pred=True, d2v=True):
	print('clearing cache files')
	if preproc:
		try:
			os.system('rm '+ TRAIN_PREPROC_CACHING_PATH)
			os.system('rm '+ TEST_PREPROC_CACHING_PATH)
		except:
			print('\nclear preproc FAILED\n')
	if tfidf:
		try:
			os.system('rm ' + DATA_PATH+'tfidf_train_reptweets.pkl')
			os.system('rm ' + DATA_PATH+'tfidf_test_reptweets.pkl')
		except:
			print('\nclear tfidf FAILED\n')
	if pred:
		try:
			os.system('rm ' + PRED_SUBMISSION_FILE)
		except:
			print('\nclear pred FAILED\n')
	if d2v:
		try:
			os.system('rm ' + DOC2VEC_MODEL_PATH)
		except:
			print('\nclear pred FAILED\n')
	print('\nclear cache completed\n')