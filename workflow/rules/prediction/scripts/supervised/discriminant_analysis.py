
def lda(X_train, X_test, y_train, y_test):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    print("Fitting LDA...")

    clf = LinearDiscriminantAnalysis()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy

def qda(X_train, X_test, y_train, y_test):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    print("Fitting QDA...")

    clf = QuadraticDiscriminantAnalysis()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy