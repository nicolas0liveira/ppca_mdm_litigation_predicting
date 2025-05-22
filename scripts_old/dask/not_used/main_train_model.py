import os
import argparse
from loguru import logger

import dask.dataframe as dd
from dask import compute

from .pipeline_train_model import (

    prepare_data_for_model,
    train_model,
    evaluate_model,
    save_artifacts,
    log_mlflow,
)

def load_parquet_data(parquet_path: str) -> dd.DataFrame:
    logger.info(f"Carregando dados de {parquet_path}")
    return dd.read_parquet(parquet_path)


def run_pipeline(parquet_path: str, target_column: str, output_dir: str):
    df = load_parquet_data(parquet_path)

    # Cria a DAG com operações atrasadas
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_model(df, target_column)
    model = train_model(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    model_path, preprocessor_path = save_artifacts(model, preprocessor, output_dir)
    log_run = log_mlflow(score, model_path, preprocessor_path)

    # Executa a DAG
    logger.info("Executando pipeline Dask...")
    compute(log_run)
    logger.success("Pipeline finalizado com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de treino com Dask + MLflow")
    parser.add_argument(
        "--execute", action="store_true", help="Executa o pipeline de treinamento"
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        required=True,
        help="Caminho para os dados em formato Parquet",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Nome da coluna alvo para predição",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Diretório onde os artefatos serão salvos",
    )

    args = parser.parse_args()

    if args.execute:
        run_pipeline(args.parquet_path, args.target_column, args.output_dir)
    else:
        logger.warning("Flag --execute não especificada. Nada será executado.")
