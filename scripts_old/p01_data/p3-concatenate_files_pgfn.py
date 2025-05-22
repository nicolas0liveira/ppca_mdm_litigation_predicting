import argparse
import os
import glob
import sys
import time
import zipfile
import re
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Parser de argumentos
parser = argparse.ArgumentParser(description="Processa os dados da PGFN e CNPJ.")
parser.add_argument("--csv", action="store_true", help="Salvar saída em CSV (padrão)")
parser.add_argument("--parquet", action="store_true", help="Salvar saída em Parquet")
args = parser.parse_args()

# Definição com base nos argumentos
SAVE_AS_CSV = args.csv or not args.parquet  # CSV é o padrão
SAVE_AS_PARQUET = args.parquet

INPUT_PGFN_DIR = "/home/nicolas/Documents/dados/pgfn/dados_para_juntar"
OUTPUT_DIR = f"{INPUT_PGFN_DIR}/concatenated/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, f"{timestamp}_alldata_pgfn_pj.csv")
OUTPUT_PARQUET_FILE = os.path.join(OUTPUT_DIR, f"{timestamp}_alldata_pgfn_pj.parquet")

CHUNK_SIZE = 100_000

pgfn_columns = [
    "CPF_CNPJ",
    "DATA_INSCRICAO",
    "INDICADOR_AJUIZADO",
    "NOME_DEVEDOR",
    "NUMERO_INSCRICAO",
    "SITUACAO_INSCRICAO",
    "TIPO_DEVEDOR",
    "TIPO_PESSOA",
    "TIPO_SITUACAO_INSCRICAO",
    "UNIDADE_RESPONSAVEL",
    "VALOR_CONSOLIDADO",
]

extra_columns = ["FILE_NAME", "ANO", "TRIMESTRE", "ORIGEM"]
all_columns = pgfn_columns + extra_columns

# Tipos
dtypes = {
    "CPF_CNPJ": "string",
    "DATA_INSCRICAO": "string",
    "INDICADOR_AJUIZADO": "string",
    "NOME_DEVEDOR": "string",
    "NUMERO_INSCRICAO": "string",
    "SITUACAO_INSCRICAO": "string",
    "TIPO_DEVEDOR": "string",
    "TIPO_PESSOA": "string",
    "TIPO_SITUACAO_INSCRICAO": "string",
    "UNIDADE_RESPONSAVEL": "string",
    "VALOR_CONSOLIDADO": "string",  # tratar dps, converti para string
}

logger.remove()
LOG_FILE = os.path.join(OUTPUT_DIR, f"{timestamp}_output.log")
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level><cyan>{level:<8}</cyan></level> | "
    "<magenta>{file.name}</magenta>:<blue>{line}</blue> - "
    "<white>{message}</white>"
)

logger.add(
    LOG_FILE,
    rotation="50 MB",
    # retention="7 days",
    encoding="utf-8",
    format=log_format,
    level="INFO",
)

logger.add(
    sys.stdout,
    format=log_format,
    colorize=True,
)

parquet_writer = None
table_schema = None
total_rows = 0
start_time = time.time()

logger.info(
    f"📁 Iniciando processamento dos arquivos PGFN...: SAVE_AS_PARQUET={SAVE_AS_PARQUET}, SAVE_AS_CSV={SAVE_AS_CSV}"
)


def extract_ano_trimestre(nome_arquivo):
    match = re.search(r"(\d{4})T([1-4])", nome_arquivo)
    return match.groups() if match else (None, None)


def extract_origem(nome_zip):
    if "FGTS.zip" in nome_zip:
        return "FGTS"
    elif "Nao_Previdenciario.zip" in nome_zip:
        return "SIDA"
    elif "Previdenciario.zip" in nome_zip:
        return "PREV"
    return "DESCONHECIDO"

# Processamento dos ZIPs
for zip_path in glob.glob(os.path.join(INPUT_PGFN_DIR, "*.zip")):
    logger.info(f"📦 ZIP: {zip_path}")
    ano, trimestre = extract_ano_trimestre(zip_path)
    origem = extract_origem(zip_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [f for f in zf.namelist() if f.endswith(".csv")]

            for csv_name in csv_files:
                logger.info(f"  📊 CSV: {csv_name}")
                try:
                    with zf.open(csv_name) as file:
                        chunk_count = 0

                        # Tentando diferentes encodings
                        for encoding in ["utf-8", "latin1", "ISO-8859-1"]:
                            try:
                                for chunk in pd.read_csv(
                                    file,
                                    chunksize=CHUNK_SIZE,
                                    encoding=encoding,
                                    sep=";",
                                    usecols=lambda x: x in pgfn_columns,
                                    dtype=dtypes,
                                ):
                                    chunk_count += 1
                                    row_count = len(chunk)
                                    total_rows += row_count
                                    logger.info(
                                        f"    🧩 Chunk {chunk_count}: {row_count} linhas"
                                    )

                                    # Preenche e ordena colunas
                                    for col in pgfn_columns:
                                        if col not in chunk.columns:
                                            chunk[col] = pd.NA
                                    chunk = chunk[pgfn_columns]

                                    # filtrar por linhas com CNPJ
                                    cd_cpf_cnpj = chunk["CPF_CNPJ"].str.replace(
                                        r"[-./]", "", regex=True
                                    )
                                    chunk["CNPJ"] = cd_cpf_cnpj.where(
                                        chunk["TIPO_PESSOA"].str.contains(
                                            "J", case=False, na=False
                                        )
                                    ).str.zfill(14)

                                    # filtra somente CNPJ e que conseguiu recuperar CNPJ (PJ)
                                    chunk = chunk[chunk["CNPJ"].notna()]

                                    # Adicionar colunas extras
                                    chunk["FILE_NAME"] = csv_name
                                    chunk["ANO"] = ano
                                    chunk["TRIMESTRE"] = trimestre
                                    chunk["ORIGEM"] = origem

                                    if SAVE_AS_CSV:
                                        header = not os.path.exists(OUTPUT_CSV_FILE)
                                        chunk.to_csv(
                                            OUTPUT_CSV_FILE,
                                            index=False,
                                            mode="a",
                                            header=header,
                                        )

                                    if SAVE_AS_PARQUET:
                                        if parquet_writer is None:
                                            table = pa.Table.from_pandas(chunk, preserve_index=False)
                                            table_schema = table.schema
                                            parquet_writer = pq.ParquetWriter(
                                                OUTPUT_PARQUET_FILE, table_schema
                                            )
                                        else:
                                            table = pa.Table.from_pandas(
                                                chunk,
                                                schema=table_schema,
                                                preserve_index=False,
                                            )
                                        parquet_writer.write_table(table)

                                break

                            except (UnicodeDecodeError, ValueError) as e:
                                file.seek(0)
                        else:
                            raise ValueError("❌ Nenhum encoding válido para leitura.")

                except Exception as e:
                    logger.exception(
                        f"❌ Erro ao processar CSV {csv_name} em {zip_path}"
                    )

    except Exception as e:
        logger.exception(f"❌ Erro ao abrir ZIP {zip_path}")

if SAVE_AS_PARQUET and parquet_writer:
    parquet_writer.close()

elapsed = time.time() - start_time
logger.success("✅ Processamento finalizado!")
logger.info(f"📊 Total de linhas processadas: {total_rows:_}")
logger.info(f"📁 Arquivos salvos em: {OUTPUT_DIR}")
hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)
time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
logger.info(f"⏱️ Tempo total: {time_str} ({elapsed:.2f} segundos)")
