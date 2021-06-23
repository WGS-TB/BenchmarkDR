def svm(X_train, y_train):
    from sklearn.svm import SVC
    print("Fitting SVM...")

    model = SVC(kernel='linear', C=0.1)
    clf = model.fit(X_train, y_train)

    return model, clf
    