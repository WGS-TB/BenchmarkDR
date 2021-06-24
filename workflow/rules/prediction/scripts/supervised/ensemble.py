def dt():
    from sklearn.tree import DecisionTreeClassifier
    print("Building DT...")

    model = DecisionTreeClassifier()

    return model

def rf():
    from sklearn.ensemble import RandomForestClassifier
    print("Building RF...")

    model = RandomForestClassifier(n_estimators=140, min_samples_split=4, bootstrap=False, max_depth=50)

    return model

def extratrees():
    from sklearn.ensemble import ExtraTreesClassifier
    print("Building ExtraTrees...")

    model = ExtraTreesClassifier()

    return model

def adaboost():
    from sklearn.ensemble import AdaBoostClassifier
    print("Building AdaBoost...")

    model = AdaBoostClassifier()

    return model

def gbt():
    from sklearn.ensemble import GradientBoostingClassifier
    print("Building GBT...")

    model = GradientBoostingClassifier()

    return model
