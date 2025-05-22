from dask_ml.linear_model import LogisticRegression
from dask_ml.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from dask import delayed, compute
from loguru import logger
from joblib import dump
import mlflow
import dask.dataframe as dd
import os


@delayed
def prepare_data_for_model(df, target_column):
    logger.info("Iniciando pré-processamento...")

    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada no DataFrame.")

    if "numero_inscricao" in df.columns:
        df = df.drop("numero_inscricao", axis=1)

    df = df.dropna()

    if df[target_column].dtype == "object":
        df[target_column] = df[target_column].map({"SIM": 1, "NÃO": 0}).astype("int")

    X = df.drop(target_column, axis=1)
    y = df[target_column].to_dask_array(lengths=True)

    cat_columns = X.select_dtypes(include="object").columns.tolist()
    num_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    logger.info(f"Colunas categóricas detectadas: {cat_columns}")

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int), cat_columns),
        ("num", StandardScaler(), num_columns)
    ])

    X_sample = X.compute()
    X_transformed = preprocessor.fit_transform(X_sample)

    X_transformed_dd = dd.from_array(
        X_transformed,
        columns=cat_columns + num_columns[:X_transformed.shape[1] - len(cat_columns)]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed_dd, y, test_size=0.2, random_state=42, shuffle=True
    )

    return X_train, X_test, y_train, y_test, preprocessor


@delayed
def train_model(X_train, y_train):
    logger.info("Treinando modelo LogisticRegression com Dask...")
    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train.to_dask_array(lengths=True), y_train)
    return model


@delayed
def evaluate_model(model, X_test, y_test):
    logger.info("Avaliando o modelo...")
    y_pred = model.predict(X_test.to_dask_array(lengths=True))
    score = accuracy_score(y_test, y_pred)
    logger.info(f"Acurácia: {score}")
    return score


@delayed
def save_artifacts(model, preprocessor, output_dir="models"):
    logger.info("Salvando modelo e pré-processador...")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    dump(model, model_path)
    dump(preprocessor, preprocessor_path)
    logger.info(f"Modelo salvo em {model_path}")
    logger.info(f"Pré-processador salvo em {preprocessor_path}")
    return model_path, preprocessor_path


@delayed
def log_mlflow(score, model_path, preprocessor_path):
    logger.info("Registrando no MLflow...")
    mlflow.set_experiment("logistic-regression-dask")
    with mlflow.start_run():
        mlflow.log_metric("accuracy", score)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(preprocessor_path)
    