import dask.dataframe as dd
from loguru import logger


from scripts.config.settings import DASK_STORAGE_OPTIONS, DATASET_S3_PATH

def load_dataset(frac: float = 1, compute: bool = False, sample: bool = False) -> dd.DataFrame:
    ddf = dd.read_parquet(DATASET_S3_PATH, storage_options=DASK_STORAGE_OPTIONS)
    logger.info(f"Dados carregados de {DATASET_S3_PATH}")

    if frac < 1:
        ddf = ddf.sample(frac=frac, random_state=42)
        logger.info(f"Amostragem de {frac*100}% dos dados")

    if compute:
        ddf = ddf.compute()
        logger.info("Dados computados")

    if sample:
        logger.info(f"Dados amostrados: {ddf.shape}")
        ddf.get_partition(0).head(1000)
    return ddf


