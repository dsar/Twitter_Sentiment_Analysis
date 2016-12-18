import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold as cross_validation_KFold

def cross_validation(clf , X_size, tfidf_train_vectors, classes, n_folds=10, random_state=4):
    """
    DESCRIPTION: 
            Given a classifier, performs k-fold cross validation on this classifier and returns the
            average score
    INPUT: 
            clf: classifier in which crossvalidation will be performed
            X_size: dimentions of train vectors
            tfidf_train_vectors: train vectors
            classes: number of classes (in our case it should always be 2 {pos, neg})
            n_folds: k-fold parameter of the cross validation method
            random_state: seed
    OUTPUT: 
            avg_test_accuracy: final average cost of cross validation
            cv: cross_validation_KFold object
    """
    cv = cross_validation_KFold(X_size, shuffle = True, n_folds=n_folds, random_state=4)
    avg_test_accuracy = np.mean(cross_val_score(clf, tfidf_train_vectors, classes, cv=cv, scoring='accuracy', verbose=True, n_jobs=-1))
    return avg_test_accuracy, cv
