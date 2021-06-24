def svm():
    from sklearn.svm import SVC
    print("Building SVM...")

    model = SVC(kernel='linear', C=0.1)

    return model
    