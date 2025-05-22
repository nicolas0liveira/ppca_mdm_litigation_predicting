import os
from dotenv import load_dotenv

# Carrega variáveis do .env, se existir
load_dotenv()

# Diretório base (normalizado para caminho absoluto)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Diretório de dados, usando variável de ambiente se existir, senão usa valor padrão
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# Subdiretórios dos dados
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")

# Outros diretórios do projeto
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")


def show_settings():
    print("📁 Configurações de Diretórios:\n")
    print(f"BASE_DIR      : {BASE_DIR}")

    print(f"DATA_DIR      : {DATA_DIR}")
    print(f"RAW_DIR       : {RAW_DIR}")
    print(f"PROCESSED_DIR : {PROCESSED_DIR}")
    print(f"INTERIM_DIR   : {INTERIM_DIR}")
    print(f"EXTERNAL_DIR  : {EXTERNAL_DIR}")

    print(f"MODELS_DIR    : {MODELS_DIR}")
    print(f"RESULTS_DIR   : {RESULTS_DIR}")
    print(f"SCRIPTS_DIR   : {SCRIPTS_DIR}")


if __name__ == "__main__":
    show_settings()
