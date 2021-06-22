def preprocess(data, label):
    import pandas as pd
    label = pd.read_csv(label)
    label.set_index(label.columns[0], inplace=True, drop=True)

    data = pd.read_csv(data, header=None)
    data.set_index(data.columns[0], inplace=True, drop=True)
    data = data.merge(label, left_index=True, right_index=True)

    labels = pd.DataFrame()

    for drug in label.columns:
        labels = pd.concat([labels, data[[drug]]], axis = 1)
        data = data.drop([drug], axis = 1)
        
    print('label set: {0}'.format(labels.shape))
    print('data set: {0}'.format(data.shape))

    return(data, labels)

def saveObject(obj, filename):
    import pickle as pickle
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)