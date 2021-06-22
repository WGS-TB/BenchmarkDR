import pandas as pd
import numpy as np
import sys, getopt
from sklearn.model_selection import train_test_split

def argCheck():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:l:", ["dfile=", "lfile="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--dfile"):
            datafile = arg
        elif opt in ("-l", "--lfile"):
            labelfile = arg
    methods = sys.argv[5:]
    print("Data file = ", datafile)
    print("Label file = ", labelfile)
    print("Running these methods = ", methods)

    return datafile, labelfile, methods

def model_run(datafile, labelfile, methods):
    from support import (preprocess, saveObject)
    from evaluation import (evaluate, CVFolds, crossValidate)
    data, labels = preprocess(data = datafile, label = labelfile)
    
    ## TODO output csv and txt with evaluation scores etc.
    df_res = pd.DataFrame([], index = labels.columns, columns = ['lr', 'lda', 'svm', 'rf', 'ingot', 'lrcv'])

    for drug in labels.columns:
        print('Analysing {0}'.format(drug))

        y = np.ravel(labels[[drug]])
        na_index = np.isnan(y)
        y = y[~na_index]

        X = data
        X = X.loc[~na_index,:]

        print(X.shape)
        print(y.shape)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

        from supervised.linear_model import lr
        model, clf = lr(X_train, y_train)
        # evaluate(clf, X_test, y_test)
        # folds = CVFolds()
        # crossValidate(model, X_train, y_train, folds)
        saveObject(model, "ml/lr.pkl")

        # from supervised.linear_model import lrcv
        # accuracy = lrcv(X_train, X_test, y_train, y_test)
        # df_res.loc[[drug], ['lrcv']] = accuracy

        # from discriminant_analysis import lda
        # accuracy = lda(X_train, X_test, y_train, y_test)
        # df_res.loc[[drug], ['lda']] = accuracy

        # accuracy = svm(X_train, X_test, y_train, y_test)
        # df_res.loc[[drug], ['svm']] = accuracy
        
        from supervised.ensemble import rf
        model, clf = rf(X_train, y_train)
        # evaluate(clf, X_test, y_test)
        # folds = CVFolds()
        # crossValidate(model, X_train, y_train, folds)
        saveObject(model, "ml/rf.pkl")

        # from supervised.ensemble import dt
        # accuracy = dt(X_train, X_test, y_train, y_test)
        
        # from supervised.ensemble import extratree
        # accuracy = extratree(X_train, X_test, y_train, y_test)

        # from supervised.ensemble import adaboost
        # accuracy = adaboost(X_train, X_test, y_train, y_test)

        # from supervised.ensemble import gbt
        # accuracy = gbt(X_train, X_test, y_train, y_test)

        # accuracy = ingot(X_train, X_test, y_train, y_test)
        # df_res.loc[[drug], ['ingot']] = accuracy

        break

datafile, labelfile, methods = argCheck()
model_run(datafile, labelfile, methods)