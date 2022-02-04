import argparse
import numpy as np
import time
import pandas as pd
import utils
import os

data_file = snakemake.input["data"]
labels_file = snakemake.input["label"]
model = snakemake.input["model"]
method = snakemake.wildcards.method
method_config_file = snakemake.input["conf"]
drug = snakemake.wildcards.drug
representation = snakemake.wildcards.representation


mode = snakemake.config["MODE"]
optimization = snakemake.config["OPTIMIZATION"]

output_file = str(snakemake.output)


def main():
    from sklearn.model_selection import train_test_split
    from skopt import BayesSearchCV
    from support import (
        preprocess,
        model_fitting,
        evaluate_classifier,
        evaluate_regression,
        save_regression_test_values,
    )

    data, labels = preprocess(data=data_file, label=labels_file)
    config_file = utils.config_reader(method_config_file)

    print("Data file = ", data_file)
    print("Mode = ", mode)
    print("Label file = ", labels_file)
    print("Method = ", method)
    print("Config file = ", method_config_file)
    print("Optimization = ", optimization)
    print("_______________________________")

    results = pd.DataFrame()

    print("Analysing {0}".format(drug))

    y = np.ravel(labels[[drug]])
    na_index = np.isnan(y)
    y = y[~na_index]

    X = data
    X = X.loc[~na_index, :]

    print("Data shape: {}".format(X.shape))
    print("Label shape: {}".format(y.shape))
    print("_______________________________")

    print("Train-Test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, **config_file["TrainTestSplit"]
    )
    print("Train data shape: {}".format(X_train.shape))
    print("Train label shape: {}".format(y_train.shape))
    print("Test data shape: {}".format(X_test.shape))
    print("Test label shape: {}".format(y_test.shape))
    print("_______________________________")

    print("Fitting training data")
    start_time = time.time()
    clf, best_parameters = model_fitting(
        drug, X_train, y_train, mode, method, model, optimization, config_file, output_file
    )
    end_time = time.time()
    print("_______________________________")

    print("Testing")
    y_pred = clf.predict(X_test)
    print("_______________________________")

    print("Evaluating")

    if mode == "Classification":
        output_file.split("csv")
        result = evaluate_classifier(y_test, y_pred)

    if mode == "MIC":
        result = evaluate_regression(y_test, y_pred)
        output_file_regression_test_values = output_file.rsplit(".")[0] + "_regression_test_values.csv")
        save_regression_test_values(y_test, y_pred, output_file_regression_test_values)

    result.insert(0, "Method", method)
    result.insert(1, "Parameters", best_parameters)
    result.insert(2, "Representation", representation)
    result.insert(3, "Drug", drug)
    result.insert(4, "Time", end_time - start_time)

    results = results.append(result)
    print("_______________________________")

    print("Saving results to {0}".format(output_file))
    results.to_csv(output_file, index=False)


main()