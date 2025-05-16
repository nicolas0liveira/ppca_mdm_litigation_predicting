import os
import argparse
import time
import dask
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from dask_ml.linear_model import LogisticRegression
from distributed import Client, LocalCluster
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from loguru import logger
from dask import delayed
import boto3
from botocore.exceptions import NoCredentialsError

load_dotenv()

# Configurações para Dask acessar dados no S3 / MinIO
DASK_STORAGE_OPTIONS = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {
        "endpoint_url": os.getenv("MINIO_ENDPOINT"),
        "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }
}

# Configurações para boto3 (upload para S3)
S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("MINIO_ENDPOINT")
S3_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

NOW = time.strftime("%Y%m%d_%H%M%S")


def upload_file_to_s3(local_path: str, s3_key: str):
    """Faz upload de arquivo local para bucket S3/MinIO"""
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        logger.info(f"Arquivo {local_path} enviado para s3://{S3_BUCKET}/{s3_key}")
    except NoCredentialsError:
        logger.error("Credenciais AWS não encontradas para upload S3.")
        raise


@delayed
def load_data(parquet_path):
    df = dd.read_parquet(parquet_path, storage_options=DASK_STORAGE_OPTIONS)
    logger.info(f"Dados carregados de {parquet_path}")
    return df


@delayed
def prepare_data_for_model(df, target_column):
    df = df.dropna()
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    logger.info("Dados pré-processados e divididos.")
    return X_train, X_test, y_train, y_test, scaler


# Funções nomeadas para extrair elementos da tupla
@delayed
def extract_X_train(data):
    return data[0]


@delayed
def extract_X_test(data):
    return data[1]


@delayed
def extract_y_train(data):
    return data[2]


@delayed
def extract_y_test(data):
    return data[3]


@delayed
def extract_scaler(data):
    return data[4]


@delayed
def train_model(X_train, y_train):
    X_train_arr = X_train.to_dask_array(lengths=True)
    y_train_arr = y_train.to_dask_array(lengths=True)

    model = LogisticRegression()
    model.fit(X_train_arr, y_train_arr)
    logger.info("Modelo LogisticRegression treinado.")
    return model


@delayed
def evaluate_model(model, X_test, y_test):
    X_test_arr = X_test.to_dask_array(lengths=True)
    y_test_arr = y_test.to_dask_array(lengths=True)

    y_pred = model.predict(X_test_arr)
    y_pred_computed = y_pred.compute()
    y_test_computed = y_test_arr.compute()

    acc = accuracy_score(y_test_computed, y_pred_computed)
    logger.info(f"Acurácia no conjunto de teste: {acc:.4f}")
    return acc


@delayed
def task_example(x):
    time.sleep(1)
    return x * x


def log_experiment(model, scaler, acc, experiment_name="dask_experiment"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        logger.info("Experimento logado no MLflow.")


def run_pipeline(parquet_s3path, target_column, experiment_name, client, execute=True):
    logger.info("Iniciando pipeline com LogisticRegression...")

    # Construção do grafo Dask
    df = load_data(parquet_s3path)
    data = prepare_data_for_model(df, target_column)

    X_train = extract_X_train(data)
    X_test = extract_X_test(data)
    y_train = extract_y_train(data)
    y_test = extract_y_test(data)
    scaler = extract_scaler(data)

    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)

    # Apenas para gerar DAG da parte delayed
    final_task = delayed(lambda *args: None, name="FinalTask")(model, scaler, acc)
    local_dag_path = f"./reports/{NOW}_pipeline_dag_logistic.svg"
    final_task.visualize(filename=local_dag_path, title="Pipeline DAG")
    logger.info(f"DAG salva localmente em {local_dag_path}")

    # Faz upload do SVG para o S3
    s3_dag_key = f"reports/{NOW}_pipeline_dag_logistic.svg"
    upload_file_to_s3(local_dag_path, s3_dag_key)

    if execute:
        # Executa e obtém resultados concretos
        model_computed, scaler_computed, acc_computed = dask.compute(model, scaler, acc)

        # Loga no MLflow com os dados concretos
        log_experiment(model_computed, scaler_computed, acc_computed, experiment_name)
        logger.info("Pipeline executado e experimento registrado com sucesso.")
    else:
        logger.info("Execução do pipeline ignorada (modo somente DAG)")

    input("Pressione Enter para encerrar...")
    client.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Dask ML com LogisticRegression")
    parser.add_argument("--execute", action="store_true", help="Executa o pipeline ao invés de apenas gerar o DAG")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        cluster = LocalCluster(dashboard_address=":9090")
        client = Client(cluster)

        logger.info("Dask Client iniciado:")
        logger.info(client)

        run_pipeline(
            parquet_s3path="s3://ppca/mdm/pgfn/processed/simple_classification.parquet",
            target_column="label",
            experiment_name="mdm_classification_experiment",
            client=client,
            execute=args.execute
        )
    except Exception as e:
        logger.exception("Erro na execução do pipeline", e)
    finally:
        if 'client' in locals():
            client.close()
        if 'cluster' in locals():
            cluster.close()
        logger.info("Cliente e cluster Dask encerrados.")
