import argparse
from joblib import load
import numpy as np
import time
import pandas as pd
import utils
import os

def evaluate(clf, X_test, y_test):
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")

def CVFolds(config_file):
    module = utils.my_import(config_file['SplitterClass']['module'])
    splitter = getattr(module, config_file['SplitterClass']['splitter'])
    folds = splitter(**config_file['SplitterClass']['params'])
    return folds

    # ## TODO argument here for each cross validation iterator
    # from sklearn.model_selection import (KFold, StratifiedKFold, RepeatedKFold, StratifiedShuffleSplit)
    # folds = StratifiedShuffleSplit(n_splits=5, random_state=8)
    # return folds

def crossValidate(clf, X_train, y_train, folds):
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=folds)
    print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))

def main():
    from sklearn.model_selection import (train_test_split, GridSearchCV, RandomizedSearchCV)
    from skopt import BayesSearchCV
    from support import (argCheck, preprocess)
    from sklearn.metrics import(balanced_accuracy_score, confusion_matrix, make_scorer)

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
        '--model', dest='modelfile', nargs='+',
        help='Path to the model file', required=True
        )
    parser.add_argument(
        '--optimize', dest='optimization',
        help='Name of optimization method', required=True,
    )
    args = parser.parse_args()

    print("Data file = ", args.datafile)
    print("Label file = ", args.labelfile)
    print("Model file(s) = ", args.modelfile)
    print("Config file = ", args.config)
    print("Optimization = ", args.optimization)
    print("_______________________________")

    data, labels = preprocess(data = args.datafile, label = args.labelfile)
    modelfiles = args.modelfile
    config_file = utils.config_reader(args.config)

    scoring = dict(Accuracy='accuracy', tp=make_scorer(utils.tp), tn=make_scorer(utils.tn),
                   fp=make_scorer(utils.fp), fn=make_scorer(utils.fn),
                   balanced_accuracy=make_scorer(balanced_accuracy_score))

    for modelfile in modelfiles:
        print("Loading ", modelfile)
        model = load(modelfile)
        modelname = modelfile.replace("ml/", "").replace(".joblib", "")

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

            if args.optimization != "None" :
                print('Hyper-parameter Tuning')
                param_grid = config_file['Models'][modelname]['cv']
                for key, value in param_grid.items():
                    if isinstance(value, str):
                        param_grid[key] = eval(param_grid[key])
                param_grid = {ele: (list(param_grid[ele])) for ele in param_grid}
                print(param_grid)
                for i in config_file['CrossValidation'].items(): print('{}: {}'.format(i[0], i[1]))

                if args.optimization == "GridSearchCV":
                    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                                scoring=scoring, **config_file['CrossValidation'])

                elif args.optimization == "RandomizedSearchCV":
                    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                scoring=scoring, **config_file['CrossValidation'])

                grid.fit(X_train, y_train)
                print('Best params: {}'.format(grid.best_params_))
                cv_results = pd.DataFrame(grid.cv_results_)
                filename = os.path.join("results", modelname + "_" + drug + ".csv")
                print('Saving results to {0}'.format(filename))
                cv_results.to_csv(filename)

                print("_______________________________")

            elif args.optimization == "None" :
                print('Not running hyper-parameter tuning')
                print('Fitting training data')
                print(model)
                start_time = time.time()
                clf = model.fit(X_train, y_train)
                end_time = time.time()
                print("_______________________________")

                print('Evaluating')
                evaluate(clf, X_test, y_test)
                folds = CVFolds(config_file)
                crossValidate(model, X_train, y_train, folds)
                
                y_pred = clf.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                sensitivity = tp/(tp + fn)
                specificity = tn/(tn + fp)

                result = dict(Drug = drug, Time= round(end_time - start_time, 2),
                    Balanced_accuracy=balanced_accuracy, tp=tp, tn=tn, fp=fp, fn=fn,
                    sensitivity = sensitivity, specificity = specificity, Model=model)
                results.append(result)
                print(result)
                print("_______________________________")
            
        print(results)
        filename = modelfile.replace('ml', 'results').replace('joblib', 'csv')
        print('Saving results to {0}'.format(filename))

        pd.DataFrame([results]).to_csv(filename)
        
        break
            

main()