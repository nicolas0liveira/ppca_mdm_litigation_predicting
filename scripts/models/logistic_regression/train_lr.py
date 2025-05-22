from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from loguru import logger
import time

from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV

from scripts.config.settings import RANDOM_STATE


def train_lr(X_train, y_train, X_test, y_test):
    # Criar um dicionário com os valores do hiperparâmetros a serem testados
    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.01, 1, 10, 100, 500, 1000],
    }

    # Configurar o GridSearchCV com 5 folds (com StratifiedKFold) e usando F1 Score como métrica de validação
    lr_model_cv = LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        lr_model_cv, param_grid, cv=kfolds, scoring="f1", verbose=2
    )

    # Treinar o modelo usando cross-validation para buscar melhores hiperparâmetros
    grid_search.fit(X_train, y_train)

    # Retornar o melhor modelo
    best_model = grid_search.best_estimator_

    # Fazer predições (classificações - 0 ou 1)
    y_pred = best_model.predict(X_test)

    # Calcular f1_score
    f1_score_value = f1_score(y_test, y_pred, zero_division=0) * 100

    return best_model, f1_score_value


def train_lr_dask(X_train, y_train, X_test, y_test, dask_client:None ):
    # client = Client("tcp://dask-scheduller:8786")
    if dask_client is None:
        logger.info("Dask Client não iniciado")
        return

    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
    }

    lr_model = LogisticRegression(solver="liblinear", random_state=42)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = DaskGridSearchCV(
        lr_model,
        param_grid,
        cv=kfolds,
        scoring="f1",
        scheduler=dask_client,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    f1_score_value = f1_score(y_test, y_pred, zero_division=0) * 100

    return best_model, f1_score_value
