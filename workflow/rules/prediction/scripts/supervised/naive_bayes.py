def gnb():
    from sklearn.naive_bayes import GaussianNB
    print("Building GaussianNB...")

    model = GaussianNB()

    return model

def cnb():
    from sklearn.naive_bayes import ComplementNB
    print("Building ComplementNB...")

    model = ComplementNB()

    return model