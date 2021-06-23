def ingot(X_train, y_train):
    # need to install package separately with v.0.05
    # pip install ingotdr pulp
    from ingot import INGOTClassifier
    print("Fitting INGOT-DR...")

    model = INGOTClassifier(lambda_p=10, lambda_z=0.01, false_positive_rate_upper_bound=0.1,
                        max_rule_size=20, solver_name='PULP_CBC_CMD')
    clf = model.fit(X_train, y_train)

    return model, clf