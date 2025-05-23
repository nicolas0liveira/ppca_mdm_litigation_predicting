from datetime import datetime
import os
import warnings

import webbrowser
import dask.dataframe as dd
from dask_ml.wrappers import ParallelPostFit
from dask_ml.preprocessing import Categorizer, StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from dask_ml.model_selection import GridSearchCV
import joblib
from dask.distributed import Client
from loguru import logger
import mlflow
from mlflow.models.signature import infer_signature
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


from scripts.config.settings import DASK_STORAGE_OPTIONS, DATASET_S3_PATH, OUTPUT_PATH, MLFLOW_TRACKING_URI


warnings.filterwarnings("ignore", category=FutureWarning, module="dask_glm")


TARGET_COL="TARGET"
COLS_TO_DROP = [TARGET_COL, "NUMERO_INSCRICAO", "CPF_CNPJ_x"]
EXPERIMENT_NAME="ppca_mdm_pgfn_classification"


def get_column_types(ddf: dd.DataFrame):
    """Retorna listas de colunas numéricas e categóricas sem computar o DataFrame"""
    dtypes = ddf.dtypes
    num_cols = dtypes[dtypes.apply(lambda x: x.kind in "if")].index.tolist()
    cat_cols = dtypes[dtypes.apply(lambda x: x == "object" or x.name == "category")].index.tolist()

    for col in COLS_TO_DROP:
        if col in num_cols:
            num_cols.remove(col)
        if col in cat_cols:
            cat_cols.remove(col)

    return num_cols, cat_cols

def split_train_test_dask(ddf: dd.DataFrame, test_size=0.3, random_state=42):
    logger.info("Removendo linhas com valores NaN (lazy)...")
    ddf = ddf.dropna()

    logger.info("Transformando coluna alvo...")
    # ddf[TARGET_COL] = ddf[TARGET_COL].str.upper().map(
    #     {"SIM": 1, "NAO": 0, "NÃO": 0},
    #     meta=(TARGET_COL, 'float64')
    # ).fillna(0)

    y = ddf[TARGET_COL]
    X = ddf.drop(columns=COLS_TO_DROP)

    logger.info("Separando treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    logger.info("Identificando tipos de colunas...")
    num_cols, cat_cols = get_column_types(X)

    logger.info("Aplicando pré-processadores com ColumnTransformer...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Categorizer(), cat_cols),
            ("num", StandardScaler(), num_cols)
        ],
        remainder="drop"
    )

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train, X_test, y_test, preprocessor, output_path):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logger.info(f"Run ID: {run.info.run_id}")
        # Preparar diretório de saída
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            output_path = os.path.join(OUTPUT_PATH, f"artefatos_{timestamp}")
            os.makedirs(output_path, exist_ok=True)
            logger.info(f"Diretório de saída: {output_path}")

        logger.info("Treinando modelo LogisticRegression (Dask)...")
        model = LogisticRegression(
            # random_state=42,
            # max_iter=100,
            # tol=1e-4,
            # # solver="admm",
            # # solver="gradient_descent",
            # solver="lbfgs",
            # penalty="l2",
            # C=1.0,
        )
        model.fit(X_train, y_train)

        # model, grid = run_grid_search(X_train, y_train, X_test, y_test, preprocessor, output_path)

        wrapper = ParallelPostFit(estimator=model)
        y_pred = wrapper.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        logger.info("Accuracy:", metrics["accuracy"])
        logger.info("F1-score:", metrics["f1_score"])
        logger.info("Precision:", metrics["precision"])
        logger.info("Recall:", metrics["recall"])
        logger.info("Confusion Matrix:\n", metrics["confusion_matrix"])
        logger.info("Classification Report:\n", metrics["classification_report"])

        y_test_np = metrics["y_test_np"]
        y_pred_np = metrics["y_pred_np"]


        # Salvar artefatos
        logger.info("Salvando artefatos e gráficos...")

        plot_confusion_matrix(y_test_np, y_pred_np, os.path.join(output_path, "confusion_matrix.png"))

        plot_roc_curve(y_test_np, y_pred_np, os.path.join(output_path, "roc_curve.png"))

        plot_precision_recall(y_test_np, y_pred_np, os.path.join(output_path, "precision_recall_curve.png"))

        plot_feature_importance(model, preprocessor, os.path.join(output_path, "feature_importances.png"))

        # Salvar modelo e preprocessor
        joblib.dump(wrapper, os.path.join(output_path, "model_wrapper.joblib"))
        joblib.dump(preprocessor, os.path.join(output_path, "preprocessor.joblib"))
        logger.success("Modelo e pré-processador salvos.")

        plot_relatorio(model, preprocessor, metrics, os.path.join(output_path, f"relatorio_{model.__class__.__name__}.txt"))

        logger.success("Treinamento finalizado.")

        # Logar experimento no MLflow
        logger.info(f"Logando experimento no MLflow: {MLFLOW_TRACKING_URI}")
        experiment_id = run.info.experiment_id
        log_experiment_mlflow(model, preprocessor, metrics, output_path, log_subruns=True, experiment_name=EXPERIMENT_NAME)
        mlflow_url = f"{MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}"
        webbrowser.open(mlflow_url)

    return wrapper


def evaluate_model(y_test, y_pred):
    y_test_np = y_test.compute() if hasattr(y_test, 'compute') else y_test
    y_pred_np = y_pred.compute() if hasattr(y_pred, 'compute') else y_pred

    acc = accuracy_score(y_test_np, y_pred_np)
    f1 = f1_score(y_test_np, y_pred_np, average='weighted')
    precision = precision_score(y_test_np, y_pred_np, average='weighted')
    recall = recall_score(y_test_np, y_pred_np, average='weighted')
    cm = confusion_matrix(y_test_np, y_pred_np)
    report = classification_report(y_test_np, y_pred_np)

    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_test_np': y_test_np,
        'y_pred_np': y_pred_np,
    }


def balance_classes_random(X, y, strategy="undersample", random_state=42):
    """
    Balanceia as classes via undersampling ou oversampling (random).
    Aceita X como ndarray ou matriz esparsa (CSR).
    """
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    max_count = counts.max()

    X_resampled = []
    y_resampled = []

    for label in classes:
        idx = np.where(y == label)[0]

        if strategy == "undersample":
            idx_sampled = resample(idx, replace=False, n_samples=min_count, random_state=random_state)
        elif strategy == "oversample":
            idx_sampled = resample(idx, replace=True, n_samples=max_count, random_state=random_state)
        else:
            raise ValueError("Estratégia inválida: use 'undersample' ou 'oversample'.")

        X_resampled.append(X[idx_sampled])
        y_resampled.append(y[idx_sampled])

    if sp.issparse(X):
        X_balanced = sp.vstack(X_resampled)
    else:
        X_balanced = np.vstack(X_resampled)

    y_balanced = np.concatenate(y_resampled)

    return X_balanced, y_balanced

def save_plot(path, plot_fn):
    plt.figure(figsize=(10, 6))
    plot_fn()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.success(f"Gráfico salvo em '{path}'.")

def plot_relatorio(model, preprocessor, metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Modelo '{model.__class__}'\n")
        f.write(f"\tAcurácia: {metrics['accuracy']:.4f}\n")
        f.write(f"\tF1-score: {metrics['f1_score']:.4f}\n")
        f.write(f"\tPrecisão: {metrics['precision']:.4f}\n")
        f.write(f"\tRecall: {metrics['recall']:.4f}\n")
        f.write(f"\n\tMatriz de confusão:\n{metrics['confusion_matrix']}\n")
        f.write(f"\nRelatório de classificação:\n{metrics['classification_report']}\n")
        f.write("\nParâmetros do modelo:\n")
        for param, value in model.get_params().items():
            f.write(f"\t{param}: {value}\n")

        f.write("\nImportância das features:\n")

        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]

        # Para modelos lineares (ex: LogisticRegression)
        if hasattr(model, "coef_"):
            importances = model.coef_.ravel()  # Flatten se for binário
        # Para modelos baseados em árvore
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = None

        if importances is not None:
            sorted_features = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)
            for name, importance in sorted_features:
                f.write(f"\t{name}: {importance:.4f}\n")
        else:
            f.write("\tEste modelo não fornece importâncias de features.\n")


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred).astype(int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NAO", "SIM"])
    def draw():
        disp.plot(cmap="Blues", values_format='d')
    save_plot(path, draw)

def plot_feature_importance(model, preprocessor, path):
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]

    # Obter importâncias
    if hasattr(model, "coef_"):
        importances = np.ravel(model.coef_)  # Flatten se for binário
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Este modelo não fornece importâncias de features.")
        return

    # Ordenar por importância
    sorted_idx = np.argsort(np.abs(importances))[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), sorted_importances)
    plt.yticks(range(len(sorted_idx)), sorted_features)
    plt.xlabel("Importância")
    plt.title("Importância das Features")
    plt.gca().invert_yaxis()  # Opcional: deixa a feature mais importante no topo
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_accuracy_series(y_true, y_pred, path):
    save_plot(path, lambda: plt.plot(y_true.index, y_true, label="Real", color="blue") or
                           plt.plot(y_pred.index, y_pred, label="Predito", color="red") or
                           plt.xlabel("Índice") or plt.ylabel("Valor") or
                           plt.title("Acurácia") or plt.legend())

def plot_roc_curve(y_true, y_pred, path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    save_plot(path, lambda: plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}") or
                           plt.plot([0, 1], [0, 1], linestyle="--", color="gray") or
                           plt.xlabel("FPR") or plt.ylabel("TPR") or
                           plt.title("Curva ROC") or plt.legend())

def plot_precision_recall(y_true, y_pred, path):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    save_plot(path, lambda: plt.plot(recall, precision) or
                           plt.xlabel("Recall") or plt.ylabel("Precision") or
                           plt.title("Curva Precision-Recall"))


def log_experiment_mlflow(model, preprocessor, metrics, artifacts_path: str, experiment_name="default", log_subruns=False):
    # Garante que está dentro de um run ativo
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("Nenhum mlflow.start_run() ativo. Certifique-se de iniciar um run antes de chamar esta função.")

    logger.info(f"Logando no run existente: {active_run.info.run_id}")
    mlflow.set_experiment(experiment_name)

    logger.info("Logando parâmetros e métricas no MLflow...")

    # GridSearchCV com subruns
    if log_subruns and hasattr(model, "cv_results_"):
        logger.info("GridSearchCV detectado — logando subexperimentos...")
        for i, params in enumerate(model.cv_results_["params"]):
            mean_score = model.cv_results_["mean_test_score"][i]
            std_score = model.cv_results_["std_test_score"][i]
            std_score_time = model.cv_results_["std_fit_time"][i]
            rank_test_score = model.cv_results_["rank_test_score"][i]

            with mlflow.start_run(run_name=f"subrun_{i}", nested=True):
                for param, value in params.items():
                    mlflow.log_param(param, value)
                mlflow.log_metric("mean_test_score", mean_score)
                mlflow.log_metric("std_test_score", std_score)
                mlflow.log_metric("std_fit_time", std_score_time)
                mlflow.log_metric("rank_test_score", rank_test_score)
                mlflow.sklearn.log_model(model, "model")
                mlflow.sklearn.log_model(preprocessor, "preprocessor")

    else:
        # Parâmetros
        for param, value in model.get_params().items():
            mlflow.log_param(param, value)

        # Métricas
        for param, value in metrics.items():
            if isinstance(value, (int, float, str)):
                mlflow.log_param(param, value)

        # Modelos
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(preprocessor, "preprocessor")

        # Artefatos
        artefatos = [
            "confusion_matrix.png",
            "roc_curve.png",
            "precision_recall_curve.png",
            f"relatorio_{model.__class__.__name__}.txt",
            "model_wrapper.joblib",
            "preprocessor.joblib",
            "feature_importances.png",
        ]

        for nome_arquivo in artefatos:
            caminho = os.path.join(artifacts_path, nome_arquivo)
            if os.path.exists(caminho):
                mlflow.log_artifact(caminho)

    logger.success("Experimento logado no MLflow com sucesso.")
    return active_run, active_run.info.run_id, active_run.info.experiment_id


# def log_experiment_mlflow(model, preprocessor, metrics, artifacts_path: str, experiment_name="default"):
#     """
#     Loga o experimento no MLflow, incluindo métricas, modelo e artefatos.
#     """
#     mlflow.set_experiment(experiment_name)

#     with mlflow.start_run():
#         logger.info("Logando parâmetros e métricas no MLflow...")

#         # Parâmetros do modelo
#         for param, value in model.get_params().items():
#             mlflow.log_param(param, value)

#         # Métricas principais
#         mlflow.log_metric("accuracy", metrics["accuracy"])
#         mlflow.log_metric("f1_score", metrics["f1_score"])
#         mlflow.log_metric("precision", metrics["precision"])
#         mlflow.log_metric("recall", metrics["recall"])

#         # Logar modelo
#         mlflow.sklearn.log_model(model, "model")
#         mlflow.sklearn.log_model(preprocessor, "preprocessor")

#         # Logar artefatos (gráficos e relatório)
#         artefatos = [
#             "confusion_matrix.png",
#             "roc_curve.png",
#             "precision_recall_curve.png",
#             f"relatorio_{model.__class__.__name__}.txt",
#             "model_wrapper.joblib",
#             "preprocessor.joblib",
#             "feature_importances.png",
#         ]

#         for nome_arquivo in artefatos:
#             caminho = os.path.join(artifacts_path, nome_arquivo)
#             if os.path.exists(caminho):
#                 mlflow.log_artifact(caminho)

#         logger.success("Experimento logado no MLflow com sucesso.")

def run_grid_search(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
    artifacts_path,
    param_grid=None,
    scoring='f1_weighted',
    cv=3,
    random_state=42,
    experiment_name=EXPERIMENT_NAME
):
    if param_grid is None:
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "max_iter": [100, 200],
            "tol": [1e-4, 1e-3]
        }

    logger.info("Executando GridSearchCV com Dask...")

    model = LogisticRegression(random_state=random_state)
    grid = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        cv=cv
    )

    grid.fit(X_train, y_train)

    # Avaliação no conjunto de teste
    y_pred = grid.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    # Logar no MLflow com subruns
    log_experiment_mlflow(
        model=grid,
        preprocessor=preprocessor,
        metrics=metrics,
        artifacts_path=artifacts_path,
        experiment_name=experiment_name,
        log_subruns=True
    )

    logger.success(f"Melhores parâmetros encontrados: {grid.best_params_}")

    logger.success(f"Melhor score: {grid.best_score_:.4f}")
    best_model = grid.best_estimator_

    return best_model, grid


def process():
    logger.info(f"Carregando dados do Parquet (Dask)...file={DATASET_S3_PATH}")
    ddf = dd.read_parquet(DATASET_S3_PATH, storage_options=DASK_STORAGE_OPTIONS, chunksize="100MB")
    ddf = ddf.repartition(partition_size="100MB")
    # ddf = ddf.head(1000, compute=True)

    logger.info("Executando pipeline de split e pré-processamento...")
    X_train, X_test, y_train, y_test, preprocessor = split_train_test_dask(ddf)

    # logger.info("Balanceando classes (undersample)...")
    # X_train, y_train = balance_classes_random(X_train, y_train, strategy="undersample")

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_path = os.path.join(OUTPUT_PATH, f"artefatos_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Diretório de saída: {output_path}")


    logger.info("Treinando modelo...")
    train_model(X_train, y_train, X_test, y_test, preprocessor,output_path)


def main():
    logger.info("Iniciando cliente Dask...")
    client = Client(n_workers=2, threads_per_worker=2, memory_limit="14GB", local_directory="./tmp/dask-worker-space")
    logger.info(f"Dask dashboard disponível em: {client.dashboard_link}")
    webbrowser.open(client.dashboard_link)

    try:
        process()
        logger.success("Pipeline finalizado com sucesso.")
        input("Pressione ENTER para encerrar e fechar o Dask Client...")

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário.")

    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
    finally:
        client.close()
        logger.info("Cliente Dask fechado.")


if __name__ == "__main__":
    main()
