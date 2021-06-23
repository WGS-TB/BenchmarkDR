def lda(X_train, y_train):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    print("Fitting LDA...")

    model = LinearDiscriminantAnalysis()
    clf = model.fit(X_train, y_train)

    return model, clf

def qda(X_train, y_train):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    print("Fitting QDA...")

    model = QuadraticDiscriminantAnalysis()
    clf = model.fit(X_train, y_train)

    return model, clf