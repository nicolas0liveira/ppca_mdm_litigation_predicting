import os
import argparse
import time
import joblib
import dask
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression
from distributed import Client, LocalCluster
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from loguru import logger
from dask import delayed
import boto3
from botocore.exceptions import NoCredentialsError
import optuna

load_dotenv()

DASK_STORAGE_OPTIONS = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {
        "endpoint_url": os.getenv("MINIO_ENDPOINT"),
        "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    },
}

S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("MINIO_ENDPOINT")
S3_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
NOW = time.strftime("%Y%m%d_%H%M%S")


def upload_file_to_s3(local_path: str, s3_key: str):
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


def save_artifacts(model, preprocessor, prefix="logistic"):
    output_dir = f"./artifacts/{NOW}_{prefix}"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    logger.info(f"Artefatos salvos em {output_dir}")
    return output_dir


@delayed
def load_data(path):
    df = dd.read_parquet(path, storage_options=DASK_STORAGE_OPTIONS)
    logger.info(f"Dados carregados de {path}")
    return df


@delayed
def prepare_data_for_model(df, target_column):
    logger.info("Iniciando pré-processamento...")

    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada no DataFrame.")

    if "numero_inscricao" in df.columns:
        df = df.drop("numero_inscricao", axis=1)

    df = df.dropna()

    # Simplificando tratamento da coluna alvo:
    if df[target_column].dtype == "object":
        # Primeiro mapear SIM/NÃO para 1/0
        df[target_column] = df[target_column].map({"SIM": 1, "NÃO": 0})
        # Caso ainda haja valores não mapeados, mapear os demais valores únicos para inteiros
        mask_na = df[target_column].isna()
        if mask_na.any().compute():
            unique_vals = df[target_column][mask_na].unique().compute().tolist()
            mapping_rest = {v: i for i, v in enumerate(unique_vals)}
            df[target_column] = df[target_column].fillna(-1).map(lambda x: mapping_rest.get(x, x) if x == -1 else x).astype("int")
        else:
            df[target_column] = df[target_column].astype("int")

    X = df.drop(target_column, axis=1)
    y = df[[target_column]]

    cat_columns = X.select_dtypes(include="object").columns.tolist()
    num_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    logger.info(f"Colunas categóricas detectadas: {cat_columns}")

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int), cat_columns),
        ("num", StandardScaler(), num_columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    return X_train, X_test, y_train, y_test, preprocessor


# Helpers para extrair elementos
@delayed
def extract_X_train(data): return data[0]

@delayed
def extract_X_test(data): return data[1]

@delayed
def extract_y_train(data): return data[2]

@delayed
def extract_y_test(data): return data[3]

@delayed
def extract_preprocessor(data): return data[4]


@delayed
def objective_delayed(params, X_train, y_train, X_test, y_test):
    C = params["C"]
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test.compute(), y_pred.compute())

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

    return acc, params


def run_study_dask(X_train, y_train, X_test, y_test, n_trials):
    study = optuna.create_study(direction="maximize")
    sampler = study.sampler

    trials = []
    for _ in range(n_trials):
        params = {
            "C": sampler.sample_float("C", 1e-4, 10.0, log=True)
        }
        trials.append(objective_delayed(params, X_train, y_train, X_test, y_test))

    results = dask.compute(*trials)

    best_acc, best_params = max(results, key=lambda x: x[0])
    return best_acc, best_params


def run_pipeline(parquet_s3path, target_column, experiment_name, client, execute=True):
    logger.info("Iniciando pipeline com LogisticRegression e Optuna...")

    df = load_data(parquet_s3path)
    data = prepare_data_for_model(df, target_column)

    X_train = extract_X_train(data)
    X_test = extract_X_test(data)
    y_train = extract_y_train(data)
    y_test = extract_y_test(data)
    preprocessor = extract_preprocessor(data)

    final_task = delayed(lambda *args: None, name="FinalTask")(X_train, X_test, y_train, y_test, preprocessor)
    dag_path = f"./reports/{NOW}_pipeline_dag_logistic.svg"
    final_task.visualize(filename=dag_path, title="Pipeline DAG")
    logger.info(f"DAG salva em {dag_path}")

    if execute:
        X_train_comp, X_test_comp, y_train_comp, y_test_comp, preprocessor_comp = dask.compute(X_train, X_test, y_train, y_test, preprocessor)
        logger.info("Dados pré-processados computados.")

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            best_acc, best_params = run_study_dask(X_train_comp, y_train_comp, X_test_comp, y_test_comp, n_trials=20)
            logger.info(f"Melhores parâmetros: {best_params}")
            logger.info(f"Melhor acurácia: {best_acc:.4f}")

            model_final = LogisticRegression(**best_params, max_iter=1000)
            model_final.fit(X_train_comp, y_train_comp)

            y_pred_final = model_final.predict(X_test_comp)
            acc_final = accuracy_score(y_test_comp.compute(), y_pred_final.compute())
            logger.info(f"Acurácia final: {acc_final:.4f}")

            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy_final", acc_final)
            mlflow.sklearn.log_model(model_final, "model_final")

            output_dir = save_artifacts(model_final, preprocessor_comp)
            logger.info(f"Artefatos salvos em {output_dir}")

    else:
        logger.info("Modo somente DAG ativo. Nenhuma execução real será feita.")

    input("Pressione Enter para encerrar...")
    client.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Dask + Optuna + MLflow")
    parser.add_argument("--execute", action="store_true", help="Executa o pipeline ao invés de apenas gerar a DAG")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cluster = LocalCluster(dashboard_address=":9090")
    client = Client(cluster)

    logger.info("Dask Client iniciado:")
    logger.info(client)

    import webbrowser
    webbrowser.open("http://localhost:9090")
    logger.info("Aguardando 5 segundos para o dashboard carregar...")
    time.sleep(5)

    try:
        run_pipeline(
            parquet_s3path="s3://ppca/mdm/pgfn/processed/dau_so_com9atributos_importantes_50NAOe50SIM.parquet",
            target_column="target",
            experiment_name="logistic_regression_optuna",
            client=client,
            execute=args.execute
        )
    except Exception as e:
        logger.error(f"Erro durante execução do pipeline: {e}")
        client.close()
