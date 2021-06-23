def dt(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    print("Fitting DT...")

    model = DecisionTreeClassifier()
    clf = model.fit(X_train, y_train)

    return model, clf

def rf(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    print("Fitting RF...")

    model = RandomForestClassifier(n_estimators=140, min_samples_split=4, bootstrap=False, max_depth=50)
    clf = model.fit(X_train, y_train)

    return model, clf

def extratree(X_train, y_train):
    from sklearn.ensemble import ExtraTreesClassifier
    print("Fitting ExtraTrees...")

    model = ExtraTreesClassifier()
    clf = model.fit(X_train, y_train)

    return model, clf

def adaboost(X_train, y_train):
    from sklearn.ensemble import AdaBoostClassifier
    print("Fitting AdaBoost...")

    model = AdaBoostClassifier()
    clf = model.fit(X_train, y_train)

    return model, clf

def gbt(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    print("Fitting GBT...")

    model = GradientBoostingClassifier()
    clf = model.fit(X_train, y_train)

    return model, clf
