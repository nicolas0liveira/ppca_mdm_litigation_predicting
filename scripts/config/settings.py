import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

RANDOM_STATE=42
TEST_SIZE=0.3

OUTPUT_PATH = os.getenv("OUTPUT_PATH", "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

DASK_STORAGE_OPTIONS = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {
        "endpoint_url": os.getenv("MINIO_ENDPOINT"),
        "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    },
}

S3_BUCKET = os.getenv("S3_BUCKET")
# S3_ENDPOINT = os.getenv("MINIO_ENDPOINT")
S3_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

DATASET_S3_PATH = os.getenv("DATASET_S3_PATH", "s3://ppca/mdm/pgfn/processed/dau_so_com9atributos_importantes_50NAOe50SIM.parquet")
# DATASET_S3_PATH = os.getenv("DATASET_S3_PATH", "s3://ppca/mdm/pgfn/processed/dau_so_com9atributos_importantes.parquet")

# DATASET_S3_PATH = os.getenv("DATASET_S3_PATH", "s3://ppca/mdm/pgfn/processed/base_modelagem_2.parquet")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))