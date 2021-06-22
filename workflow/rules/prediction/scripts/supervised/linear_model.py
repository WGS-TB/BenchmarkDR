def lr(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    print("Fitting LR...")

    model = LogisticRegression(penalty='l2', solver='newton-cg')
    clf = model.fit(X_train, y_train)

    return model, clf

def lrcv(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegressionCV
    
    print("Fitting LR CV...")

    cv = crossValidation()

    clf = LogisticRegressionCV(cv=cv, penalty='l2', solver='newton-cg')

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy

def sgd(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import SGDClassifier
    print("Fitting SGD...")

    clf = SGDClassifier()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy
