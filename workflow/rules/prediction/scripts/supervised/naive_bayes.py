def gnb(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    print("Fitting GaussianNB...")

    model = GaussianNB()
    clf = model.fit(X_train, y_train)

    return model, clf

def cnb(X_train, y_train):
    from sklearn.naive_bayes import ComplementNB
    print("Fitting ComplementNB...")

    model = ComplementNB()
    clf = model.fit(X_train, y_train)

    return model, clf