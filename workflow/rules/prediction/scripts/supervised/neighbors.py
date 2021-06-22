def knn(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    print("Fitting KNN...")

    clf = KNeighborsClassifier()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy

def ncc(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import NearestCentroid
    print("Fitting NearestCentroid...")

    clf = NearestCentroid()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy