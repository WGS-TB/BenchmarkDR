def lr(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    print("Fitting LR...")

    model = LogisticRegression(penalty='l2', solver='newton-cg')
    clf = model.fit(X_train, y_train)

    return model, clf

def lrcv(X_train, y_train):
    from sklearn.linear_model import LogisticRegressionCV
    
    print("Fitting LR CV...")

    cv = crossValidation()

    model = LogisticRegressionCV(cv=cv, penalty='l2', solver='newton-cg')
    clf = model.fit(X_train, y_train)

    return model, clf

def sgd(X_train, y_train):
    from sklearn.linear_model import SGDClassifier
    print("Fitting SGD...")

    model = SGDClassifier()
    clf = model.fit(X_train, y_train)

    return model, clf
