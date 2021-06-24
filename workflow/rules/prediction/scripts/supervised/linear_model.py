def lr():
    from sklearn.linear_model import LogisticRegression
    print("Building LR...")

    model = LogisticRegression(penalty='l2', solver='newton-cg')

    return model

def lrcv():
    from sklearn.linear_model import LogisticRegressionCV
    
    print("Building LR CV...")

    cv = crossValidation()

    model = LogisticRegressionCV(cv=cv, penalty='l2', solver='newton-cg')
    clf = model.fit()

    return model, clf

def sgd():
    from sklearn.linear_model import SGDClassifier
    print("Building SGD...")

    model = SGDClassifier()

    return model
