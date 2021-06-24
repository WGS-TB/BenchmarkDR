def knn():
    from sklearn.neighbors import KNeighborsClassifier
    print("Building KNN...")

    model = KNeighborsClassifier()

    return model

def ncc():
    from sklearn.neighbors import NearestCentroid
    print("Building NearestCentroid...")

    model = NearestCentroid()

    return model