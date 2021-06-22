def gnb(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    print("Fitting GaussianNB...")

    clf = GaussianNB(C=0.1, penalty='l2', solver='newton-cg')

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy

def cnb(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import ComplementNB
    print("Fitting ComplementNB...")

    clf = ComplementNB(C=0.1, penalty='l2', solver='newton-cg')

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy