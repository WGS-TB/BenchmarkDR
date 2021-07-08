import pandas as pd
import numpy as np
import sys, getopt
from sklearn.model_selection import train_test_split
from support import argCheck

def model_run(datafile, labelfile, methods):
    from support import (preprocess, saveObject)
    from evaluation import (evaluate, CVFolds, crossValidate)
    data, labels = preprocess(data = datafile, label = labelfile)
    
    for drug in labels.columns:
        print('Analysing {0}'.format(drug))

        y = np.ravel(labels[[drug]])
        na_index = np.isnan(y)
        y = y[~na_index]

        X = data
        X = X.loc[~na_index,:]
        # X = np.sign(X)

        print(X.shape)
        print(y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

        if "lr" in methods:
            from supervised.linear_model import lr
            model, clf = lr(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/lr.pkl")

        if "lr" in methods:
            from supervised.linear_model import sgd
            model, clf = sgd(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/sgd.pkl")

        if "lda" in methods:
            from supervised.discriminant_analysis import lda
            model, clf = lda(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/lda.pkl")

        if "qda" in methods:
            from supervised.discriminant_analysis import qda
            model, clf = qda(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/qda.pkl")
        
        if "rf" in methods:
            from supervised.ensemble import rf
            model, clf = rf(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/rf.pkl")

        if "dt" in methods:
            from supervised.ensemble import dt
            model, clf = dt(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/dt.pkl")

        if "et" in methods:
            from supervised.ensemble import extratrees
            model, clf = extratrees(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/et.pkl")

        if "adb" in methods:
            from supervised.ensemble import adaboost
            model, clf = adaboost(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/adb.pkl")

        if "gbt" in methods:
            from supervised.ensemble import gbt
            model, clf = gbt(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/gbt.pkl")

        if "svm" in methods:
            from supervised.svm import svm
            model, clf = svm(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/svm.pkl")

        if "gnb" in methods:
            from supervised.naive_bayes import gnb
            model, clf = gnb(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/gnb.pkl")

        if "cnb" in methods:
            from supervised.naive_bayes import cnb
            model, clf = cnb(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/cnb.pkl")

        if "knn" in methods:
            from supervised.neighbors import knn
            model, clf = knn(X_train, y_train)
            evaluate(clf, X_test, y_test)
            folds = CVFolds()
            crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/knn.pkl")
        
        if "ncc" in methods:
            from supervised.neighbors import ncc
            model, clf = ncc(X_train, y_train)
            evaluate(clf, X_test, y_test)
            # folds = CVFolds()
            # crossValidate(model, X_train, y_train, folds)
            saveObject(model, "ml/ncc.pkl")

        # from supervised.specials import ingot
        # model, clf = ingot(X_train, y_train)
        # evaluate(clf, X_test, y_test)
        # folds = CVFolds()
        # crossValidate(model, X_train, y_train, folds)
        # saveObject(model, "ml/ingot.pkl")

        break

datafile, labelfile, methods = argCheck()
model_run(datafile, labelfile, methods)