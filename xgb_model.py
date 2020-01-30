from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def xgb_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

    model = XGBClassifier(max_depth=7, random_state=seed, min_child_weight=15, learning_rate=0.001,
                          n_estimators=400)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)

    # evaluate predictions
    print("XGB Train accuracy: %.2f%%" % (accuracy_score(y_train, y_pred_train) * 100.0))
    print("XGB Test accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100.0))
    
    return model
