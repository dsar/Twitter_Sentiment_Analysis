import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold as cross_validation_KFold

def cross_validation(clf , X_size, tfidf_train_vectors, classes, n_folds=10, random_state=4):
    cv = cross_validation_KFold(X_size, shuffle = True, n_folds=n_folds, random_state=4)
    avg_test_accuracy = np.mean(cross_val_score(clf, tfidf_train_vectors, classes, cv=cv, scoring='accuracy'))
    return avg_test_accuracy, cv