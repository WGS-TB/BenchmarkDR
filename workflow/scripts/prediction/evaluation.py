import argparse
import numpy as np
import time
import pandas as pd
import utils
import os

def main():
    from sklearn.model_selection import train_test_split
    from skopt import BayesSearchCV
    from support import (preprocess, model_fitting, evaluate_classifier, evaluate_regression)

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
    parser.add_argument(
        '--mode', dest='mode',
        help='Mode of Labels', required=True
    )

    args = parser.parse_args()

    data, labels = preprocess(data = args.datafile, label = args.labelfile)
    config_file = utils.config_reader(args.config)

    print("Data file = ", data)
    print("Mode = ", args.mode)
    print("Label file = ", labels)
    print("Model = ", args.modelname)
    print("Config file = ", config_file)
    print("Optimization = ", args.optimization)
    print("_______________________________")

    
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

        if args.mode == "Classification":
          result = evaluate_classifier(y_test, y_pred)
        
        if args.mode == "MIC":
          result = evaluate_regression(y_test, y_pred)

        result.insert(0, "Drug", drug)
        result.insert(1, "Time", end_time-start_time)
        results = results.append(result)
        print("_______________________________")

    print('Saving results to {0}'.format(args.outfile))
    results.to_csv(args.outfile, index =False)
            
main()