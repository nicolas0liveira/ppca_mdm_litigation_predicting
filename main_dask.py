from typing import Tuple
import webbrowser
import dask.dataframe as dd
import dask.array as da
from dask_ml.preprocessing import Categorizer, StandardScaler
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from dask_ml.metrics import accuracy_score
from dask.distributed import Client
from loguru import logger
from dask_ml.wrappers import ParallelPostFit

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dask_glm")

from scripts.config.settings import DASK_STORAGE_OPTIONS, DATASET_S3_PATH


def get_column_types(ddf: dd.DataFrame):
    """Retorna listas de colunas numéricas e categóricas sem computar o DataFrame"""
    dtypes = ddf.dtypes
    num_cols = dtypes[dtypes.apply(lambda x: x.kind in "if")].index.tolist()
    cat_cols = dtypes[dtypes.apply(lambda x: x == "object" or x.name == "category")].index.tolist()

    # Remoção de colunas que não devem ser incluídas
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
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    logger.info("Identificando tipos de colunas...")
    num_cols, cat_cols = get_column_types(X)

    logger.info("Aplicando Categorizer e StandardScaler...")
    categorizer = Categorizer()
    X_train_df[cat_cols] = categorizer.fit_transform(X_train_df[cat_cols])
    X_test_df[cat_cols] = categorizer.transform(X_test_df[cat_cols])

    scaler = StandardScaler()
    X_train_df[num_cols] = scaler.fit_transform(X_train_df[num_cols])
    X_test_df[num_cols] = scaler.transform(X_test_df[num_cols])

    def df_to_dask_array(df, cols):
        arrays = [df[col].to_dask_array(lengths=True).reshape(-1, 1) for col in cols]
        return da.hstack(arrays)

    X_train = df_to_dask_array(X_train_df, cat_cols + num_cols)
    X_test = df_to_dask_array(X_test_df, cat_cols + num_cols)
    y_train = y_train.to_dask_array(lengths=True)
    y_test = y_test.to_dask_array(lengths=True)

    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train, X_test, y_test):
    logger.info("Treinando modelo LogisticRegression (Dask)...")
    print(X_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    logger.info("Avaliando modelo...")
    wrapper = ParallelPostFit(estimator=model)
    y_pred = wrapper.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.success(f"Acurácia: {acc:.4f}")
    return wrapper


def process():
    logger.info("Carregando dados do Parquet (Dask)...")
    ddf = dd.read_parquet(DATASET_S3_PATH, storage_options=DASK_STORAGE_OPTIONS)
    ddf = ddf.repartition(partition_size="500MB")  # ou npartitions=20
    ddf = ddf.persist()


    logger.info("Executando pipeline de split e pré-processamento...")
    X_train, X_test, y_train, y_test = split_train_test_dask(ddf)

    logger.info("Treinando modelo...")
    model = train_model(X_train, y_train, X_test, y_test)

    return model


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
