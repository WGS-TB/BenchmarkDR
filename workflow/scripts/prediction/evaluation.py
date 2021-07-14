import argparse
import numpy as np
import time
import pandas as pd
import utils
import os

def crossValidate(clf, X_train, y_train, folds):
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=folds)
    print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))

def main():
    from sklearn.model_selection import train_test_split
    from skopt import BayesSearchCV
    from support import (preprocess, model_fitting, evaluate)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', dest='datafile', metavar='FILE',
        help='Path to the data file', required=True,
    )
    parser.add_argument(
        '--label', dest='labelfile', metavar='FILE',
        help='Path to the label file', required=True,
    )
    parser.add_argument(
        '--config', dest='config', metavar='FILE',
        help='Path to the config file', required=True,
    )
    parser.add_argument(
        '--model', dest='modelname',
        help='Name of model', required=True
    )
    parser.add_argument(
        '--modelfile', dest='modelfile',metavar='FILE',
        help='Path to the model file', required=True
    )
    parser.add_argument(
        '--optimize', dest='optimization',
        help='Name of optimization method', required=True,
    )
    parser.add_argument(
        '--outfile', dest='outfile',
        help='Name of output file', required=True
    )
    args = parser.parse_args()

    print("Data file = ", args.datafile)
    print("Label file = ", args.labelfile)
    print("Model = ", args.modelname)
    print("Config file = ", args.config)
    print("Optimization = ", args.optimization)
    print("_______________________________")

    data, labels = preprocess(data = args.datafile, label = args.labelfile)
    config_file = utils.config_reader(args.config)

    
    results=pd.DataFrame()

    for drug in labels.columns:
        print('Analysing {0}'.format(drug))

        y = np.ravel(labels[[drug]])
        na_index = np.isnan(y)
        y = y[~na_index]

        X = data
        X = X.loc[~na_index,:]

        print('Data shape: {}'.format(X.shape))
        print('Label shape: {}'.format(y.shape))
        print("_______________________________")

        print('Train-Test split')
        X_train, X_test, y_train, y_test = train_test_split(X, y, **config_file['TrainTestSplit'])
        print('Train data shape: {}'.format(X_train.shape))
        print('Train label shape: {}'.format(y_train.shape))
        print('Test data shape: {}'.format(X_test.shape))
        print('Test label shape: {}'.format(y_test.shape))
        print("_______________________________")

        print('Fitting training data')
        start_time = time.time()
        clf = model_fitting(args, drug, X_train, y_train, config_file)
        end_time = time.time()
        print("_______________________________")

        print('Testing')
        y_pred = clf.predict(X_test)
        print("_______________________________")

        print('Evaluating')
        result = evaluate(y_test, y_pred)
        result.insert(0, "Drug", drug)
        result.insert(1, "Time", end_time-start_time)
        results = results.append(result)
        print("_______________________________")

    print('Saving results to {0}'.format(args.outfile))
    results.to_csv(args.outfile, index =False)
            
main()