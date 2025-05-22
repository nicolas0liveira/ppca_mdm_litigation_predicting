from sklearn.metrics import f1_score, log_loss

def make_predictions(X_train, X_test, y_train, y_test, model):
    y_pred = model.predict(X_test)

    f1_score_value = f1_score(y_test, y_pred) * 100
    bce = log_loss(y_test, y_pred)

    return f1_score_value, bce