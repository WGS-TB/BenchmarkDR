def svm(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    print("Fitting SVM...")

    clf = SVC(kernel='linear', C=0.1)

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy