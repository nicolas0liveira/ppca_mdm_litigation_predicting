
import os
import time
import boto3
from botocore.exceptions import NoCredentialsError

from settings import S3_BUCKET, S3_ENDPOINT, S3_KEY, S3_SECRET
import joblib
from loguru import logger

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