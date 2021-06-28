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
    from joblib import load
    import numpy as np
    from sklearn.model_selection import train_test_split
    from support import (argCheck, preprocess)
    import time
    import pandas as pd
    from sklearn.metrics import(balanced_accuracy_score, confusion_matrix)

    datafile, labelfile, modelfile = argCheck()
    data, labels = preprocess(data = datafile, label = labelfile)
    model = load(modelfile)

    results=[]

    for drug in labels.columns:
        print('Analysing {0}'.format(drug))

        y = np.ravel(labels[[drug]])
        na_index = np.isnan(y)
        y = y[~na_index]

        X = data
        X = X.loc[~na_index,:]
        # X = np.sign(X)

        print('Data shape: {}'.format(X.shape))
        print('Label shape: {}'.format(y.shape))
        print("_______________________________")

        
        print('Train-Test split')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
        print('Train data shape: {}'.format(X_train.shape))
        print('Train label shape: {}'.format(y_train.shape))
        print('Test data shape: {}'.format(X_test.shape))
        print('Test label shape: {}'.format(y_test.shape))
        print("_______________________________")


        print('Fitting')
        start_time = time.time()
        clf = model.fit(X_train, y_train)
        print("_______________________________")


        print('Evaluating')
        evaluate(clf, X_test, y_test)
        folds = CVFolds()
        crossValidate(model, X_train, y_train, folds)
        y_pred = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        sensitivity = tp/(tp + fn)
        specificity = tn/(tn + fp)
        end_time = time.time()

        result = dict(Drug = drug, Time= round(end_time - start_time, 2),
            Balanced_accuracy=balanced_accuracy, tp=tp, tn=tn, fp=fp, fn=fn,
            sensitivity = sensitivity, specificity = specificity, Model=model)
        results.append(result)
        print(result)
        print("_______________________________")

    print(results)
    filename = modelfile.replace('ml', 'results')
    filename = filename.replace('joblib', 'csv')
    print('Saving results to {0}'.format(filename))

    pd.DataFrame([results]).to_csv(filename)
        

main()