def dt(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    print("Fitting DT...")

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy

def rf(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    print("Fitting RF...")

    clf = RandomForestClassifier(n_estimators=140, min_samples_split=4, bootstrap=False, max_depth=50)

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy

def extratree(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import ExtraTreesClassifier
    print("Fitting ExtraTrees...")

    clf = ExtraTreesClassifier()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy

def adaboost(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    print("Fitting AdaBoost...")

    clf = AdaBoostClassifier()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy

def gbt(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    print("Fitting GBT...")

    clf = GradientBoostingClassifier()

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")
    return accuracy
