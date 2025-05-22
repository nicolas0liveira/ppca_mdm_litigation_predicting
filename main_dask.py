from datetime import datetime
import os
import warnings
import webbrowser
import dask.dataframe as dd
from dask_ml.wrappers import ParallelPostFit
from dask_ml.preprocessing import Categorizer, StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
import joblib
from dask.distributed import Client
from loguru import logger

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


from scripts.config.settings import DASK_STORAGE_OPTIONS, DATASET_S3_PATH, OUTPUT_PATH

warnings.filterwarnings("ignore", category=FutureWarning, module="dask_glm")

def get_column_types(ddf: dd.DataFrame):
    """Retorna listas de colunas numéricas e categóricas sem computar o DataFrame"""
    dtypes = ddf.dtypes
    num_cols = dtypes[dtypes.apply(lambda x: x.kind in "if")].index.tolist()
    cat_cols = dtypes[dtypes.apply(lambda x: x == "object" or x.name == "category")].index.tolist()

    for col in ["target", "numero_inscricao"]:
        if col in num_cols:
            num_cols.remove(col)
        if col in cat_cols:
            cat_cols.remove(col)

    return num_cols, cat_cols

def split_train_test_dask(ddf: dd.DataFrame, test_size=0.3, random_state=42):
    logger.info("Removendo linhas com valores NaN (lazy)...")
    ddf = ddf.dropna()

    logger.info("Transformando coluna alvo...")
    ddf["target"] = ddf["target"].str.upper().map(
        {"SIM": 1, "NAO": 0, "NÃO": 0},
        meta=('target', 'float64')
    ).fillna(0)

    y = ddf["target"]
    X = ddf.drop(columns=["target", "numero_inscricao"])

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

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_model(X_train, y_train, X_test, y_test, preprocessor):
    logger.info("Treinando modelo LogisticRegression (Dask)...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

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

    # Preparar diretório de saída
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    output_path = os.path.join(OUTPUT_PATH, f"artefatos_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Diretório de saída: {output_path}")

    # Salvar artefatos
    logger.info("Salvando artefatos e gráficos...")

    plot_confusion_matrix(y_test_np, y_pred_np, os.path.join(output_path, "confusion_matrix.png"))

    plot_roc_curve(y_test_np, y_pred_np, os.path.join(output_path, "roc_curve.png"))

    plot_precision_recall(y_test_np, y_pred_np, os.path.join(output_path, "precision_recall_curve.png"))

    # plot_feature_importance(model, feature_names, os.path.join(output_path, "feature_importances.png"))

    # Salvar modelo e preprocessor
    joblib.dump(wrapper, os.path.join(output_path, "model_wrapper.joblib"))
    joblib.dump(preprocessor, os.path.join(output_path, "preprocessor.joblib"))
    logger.success("Modelo e pré-processador salvos.")

    plot_relatorio(model, preprocessor, metrics, os.path.join(output_path, f"relatorio_{model.__class__.__name__}.txt"))

    logger.success("Treinamento finalizado.")

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
        f.write(f"\nParâmetros do modelo:\n")
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
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NAO", "SIM"])
    save_plot(path, lambda: disp.plot())

def plot_feature_importance(model, feature_names, path):
    importances = model.coef_[0]
    sorted_idx = importances.argsort()[::-1]
    save_plot(path, lambda: plt.barh(range(len(sorted_idx)), importances[sorted_idx]) or
                           plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx]) or
                           plt.xlabel("Importância") or plt.title("Importância das Features"))

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

def process():
    logger.info("Carregando dados do Parquet (Dask)...")
    ddf = dd.read_parquet(DATASET_S3_PATH, storage_options=DASK_STORAGE_OPTIONS, chunksize="100MB")
    ddf = ddf.repartition(partition_size="100MB")

    logger.info("Executando pipeline de split e pré-processamento...")
    X_train, X_test, y_train, y_test, preprocessor = split_train_test_dask(ddf)

    logger.info("Treinando modelo...")
    train_model(X_train, y_train, X_test, y_test, preprocessor)


def main():
    logger.info("Iniciando cliente Dask...")
    client = Client(n_workers=2, threads_per_worker=2, memory_limit="14GB", local_directory="./tmp/dask-worker-space")
    logger.info(f"Dask dashboard disponível em: {client.dashboard_link}")
    # webbrowser.open(client.dashboard_link)

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
