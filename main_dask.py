from typing import Tuple
import webbrowser
import dask.dataframe as dd
from dask_ml.preprocessing import Categorizer, StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from dask_ml.metrics import accuracy_score
from dask.distributed import Client
from loguru import logger
from dask_ml.wrappers import ParallelPostFit
from sklearn.compose import ColumnTransformer
import warnings

from scripts.config.settings import DASK_STORAGE_OPTIONS, DATASET_S3_PATH

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
    logger.info("Transformando coluna alvo...")
    ddf["target"] = ddf["target"].str.upper().map(
        {"SIM": 1, "NAO": 0, "NÃO": 0},
        meta=('target', 'float64')
    )

    y = ddf["target"]
    X = ddf.drop(columns=["target", "numero_inscricao"])

    logger.info("Separando treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False
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



def train_model(X_train, y_train, X_test, y_test):
    logger.info("Treinando modelo LogisticRegression (Dask)...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    logger.info("Avaliando modelo...")
    wrapper = ParallelPostFit(estimator=model)  # Aqui sim é o lugar correto
    y_pred = wrapper.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.success(f"Acurácia: {acc:.4f}")
    return wrapper



def process():
    logger.info("Carregando dados do Parquet (Dask)...")
    ddf = dd.read_parquet(DATASET_S3_PATH, storage_options=DASK_STORAGE_OPTIONS)
    ddf = ddf.repartition(partition_size="100MB")

    logger.info("Executando pipeline de split e pré-processamento...")
    X_train, X_test, y_train, y_test, preprocessor = split_train_test_dask(ddf)

    logger.info("Treinando modelo...")
    model = train_model(X_train, y_train, X_test, y_test)

    return model, preprocessor

def main():
    logger.info("Iniciando cliente Dask...")
    client = Client()
    logger.info(f"Dask dashboard disponível em: {client.dashboard_link}")
    webbrowser.open(client.dashboard_link)

    try:
        model = process()
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
