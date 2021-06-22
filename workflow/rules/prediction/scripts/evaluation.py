from scipy.sparse.construct import random


def evaluate(clf, X_test, y_test):
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")

def CVFolds():
    ## TODO argument here for each cross validation iterator
    from sklearn.model_selection import (KFold, StratifiedKFold, RepeatedKFold, StratifiedShuffleSplit)
    folds = StratifiedShuffleSplit(n_splits=5, random_state=8)
    return folds

def crossValidate(clf, X_train, y_train, folds):
    import numpy as np
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=folds)
    print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))