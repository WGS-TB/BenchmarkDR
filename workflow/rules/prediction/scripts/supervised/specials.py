def ingot(X_train, X_test, y_train, y_test):
    # need to install package separately with v.0.05
    # pip install ingotdr pulp
    from ingot import INGOTClassifier
    print("Fitting INGOT-DR...")
    clf = INGOTClassifier(lambda_p=10, lambda_z=0.01, false_positive_rate_upper_bound=0.1,
                        max_rule_size=20, solver_name='PULP_CBC_CMD')
    #clf = ingot.INGOTClassifier(solver_name='PULP_CBC_CMD', solver_options={'tmpDir': 'Fernando/Awesome/Directory'}) TRY THIS

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    print("_______________________________")

    return accuracy
