def knn(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    print("Fitting KNN...")

    model = KNeighborsClassifier()
    clf = model.fit(X_train, y_train)

    return model, clf

def ncc(X_train, y_train):
    from sklearn.neighbors import NearestCentroid
    print("Fitting NearestCentroid...")

    model = NearestCentroid()
    clf = model.fit(X_train, y_train)

    return model, clf