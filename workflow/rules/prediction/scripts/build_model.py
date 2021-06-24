import pandas as pd
import numpy as np
import sys

def build_model(methods):
    from support import saveObject

    if "lr" in methods:
        from supervised.linear_model import lr
        model = lr()
        saveObject(model, "ml/lr.pkl")

    if "sgd" in methods:
        from supervised.linear_model import sgd
        model = sgd()
        saveObject(model, "ml/sgd.pkl")

    if "lda" in methods:
        from supervised.discriminant_analysis import lda
        model = lda()
        saveObject(model, "ml/lda.pkl")

    if "qda" in methods:
        from supervised.discriminant_analysis import qda
        model = qda()
        saveObject(model, "ml/qda.pkl")
    
    if "rf" in methods:
        from supervised.ensemble import rf
        model = rf()
        saveObject(model, "ml/rf.pkl")

    if "dt" in methods:
        from supervised.ensemble import dt
        model = dt()
        saveObject(model, "ml/dt.pkl")

    if "et" in methods:
        from supervised.ensemble import extratrees
        model = extratrees()
        saveObject(model, "ml/et.pkl")

    if "adb" in methods:
        from supervised.ensemble import adaboost
        model = adaboost()
        saveObject(model, "ml/adb.pkl")

    if "gbt" in methods:
        from supervised.ensemble import gbt
        model = gbt()
        saveObject(model, "ml/gbt.pkl")

    if "svm" in methods:
        from supervised.svm import svm
        model = svm()
        saveObject(model, "ml/svm.pkl")

    if "gnb" in methods:
        from supervised.naive_bayes import gnb
        model = gnb()
        saveObject(model, "ml/gnb.pkl")

    if "cnb" in methods:
        from supervised.naive_bayes import cnb
        model = cnb()
        saveObject(model, "ml/cnb.pkl")

    if "knn" in methods:
        from supervised.neighbors import knn
        model = knn()
        saveObject(model, "ml/knn.pkl")
    
    # if "ncc" in methods:
    #     from supervised.neighbors import ncc
    #     model = ncc()
    #     evaluate(clf, X_test, y_test)
    #     # folds = CVFolds()
    #     # crossValidate(model, , folds)
    #     saveObject(model, "ml/ncc.pkl")

    # from supervised.specials import ingot
    # model = ingot()
    # evaluate(clf, X_test, y_test)
    # folds = CVFolds()
    # crossValidate(model, , folds)
    # saveObject(model, "ml/ingot.pkl")

methods = sys.argv[1:]
print(methods)
build_model(methods)