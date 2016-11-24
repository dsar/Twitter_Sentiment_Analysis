import csv
from options import *

def create_csv_submission(y_pred):
    with open(DATA_PATH+PRED_SUBMISSION_FILE, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        r1 = 1
        for r2 in y_pred:
            if r2 == 'pos':
                writer.writerow({'Id':int(r1),'Prediction':1})
            elif r2 == 'neg':
                writer.writerow({'Id':int(r1),'Prediction':-1})
            r1+=1
