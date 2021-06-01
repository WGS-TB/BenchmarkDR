import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

res = []

def preprocess(data, label):
    label = pd.read_csv(label)
    label.set_index(label.columns[0], inplace=True, drop=True)

    data = pd.read_csv(data, header=None)
    data.set_index(data.columns[0], inplace=True, drop=True)
    data = data.merge(label, left_index=True, right_index=True)

    labels = []
    for drug in label.columns:
        labels.append(data[[drug]])
        data = data.drop([drug], axis = 1)

    print('label set has {0} lists of length {1}'.format(len(labels), len(labels[0])))
    print('data set: {0}'.format(data.shape))

    return(data, labels)

def lr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    from sklearn.linear_model import LogisticRegression
    lr_model_linear = LogisticRegression(C=0.1, penalty='l2', solver='newton-cg').fit(X_train, y_train)
    accuracy = lr_model_linear.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")

    return accuracy

def model_run():
    global res
    data, labels = preprocess(data = snakemake.input[0], label = snakemake.input[1])

    for i in range(0, len(labels)):
        print('{0} out of {1}'.format(i+1, len(labels)))
        print(labels[i].columns)
        y = np.ravel(labels[i])
        na_index = np.isnan(y)
        y = y[~na_index]

        X = data
        X = X.loc[~na_index,:]

        print(X.shape)
        print(y.shape)
        accuracy = lr(X, y)
        res.append(labels[i].columns)
        res.append(accuracy)

    f = open(snakemake.output[0], 'w')
    f.write(str(res) + '\n')
    f.close()

model_run()