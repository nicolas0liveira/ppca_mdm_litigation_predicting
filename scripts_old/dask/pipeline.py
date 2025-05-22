import os
import argparse
import time
import joblib
from loguru import logger
import boto3
import optuna

if __name__ == "__main__":
    args = parse_args()


logger.info("Iniciando pipeline com LogisticRegression e Optuna...")

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