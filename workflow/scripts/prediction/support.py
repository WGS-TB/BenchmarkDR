def argCheck():
    import getopt
    import sys
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:l:m:", ["dfile=", "lfile=", "mfile="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--dfile"):
            datafile = arg
        elif opt in ("-l", "--lfile"):
            labelfile = arg
        elif opt in ("-m", "--mfile"):
            modelfile = arg
    print("Data file = ", datafile)
    print("Label file = ", labelfile)
    print("Running model = ", modelfile)
    print("_______________________________")


    return datafile, labelfile, modelfile

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
    print("_______________________________")


    return(data, labels)

def saveObject(obj, filename):
    import pickle as pickle
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)