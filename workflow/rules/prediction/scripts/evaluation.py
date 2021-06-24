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

def main():
    import pickle as pk
    import numpy as np
    from sklearn.model_selection import train_test_split
    from support import (argCheck, preprocess)
    
    datafile, labelfile, modelfile = argCheck()
    data, labels = preprocess(data = datafile, label = labelfile)
    modelfile = open(modelfile, "rb")
    model = pk.load(modelfile)

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
        clf = model.fit(X_train, y_train)
        evaluate(clf, X_test, y_test)
        folds = CVFolds()
        crossValidate(model, X_train, y_train, folds)
        
        break

main()