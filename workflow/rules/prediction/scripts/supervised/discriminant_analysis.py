def lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    print("Building LDA...")

    model = LinearDiscriminantAnalysis()

    return model

def qda():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    print("Building QDA...")

    model = QuadraticDiscriminantAnalysis()

    return model