import os
import glob
import zipfile
import pandas as pd

from scripts.config.settings import BASE_DIR


INPUT_DIR = os.path.join(BASE_DIR, "../data/raw/pgfn/")

# INPUT_DIR = "/home/nicolas/Documents/dados/pgfn/data/raw/pgfn/"

INPUT_DIR = "/home/nicolas/Documents/dados/pgfn/dados_para_juntar"
OUTPUT_CSV = f"{INPUT_DIR}/columns_matrix.csv"
OUTPUT_COLUNAS_COMUNS_CSV = f"{INPUT_DIR}/columns_common_all_files.csv"

arquivo_colunas = {}
todas_colunas = set()

colunas_comuns = None


def ler_primeira_linha_seguro(file):
    for encoding in ["utf-8", "latin1"]:
        try:
            file.seek(0)
            linha = file.readline().decode(encoding).strip()
            return linha
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "Não foi possível decodificar o arquivo com utf-8 nem latin1."
    )


# Percorrer todos os ZIPs
for zip_path in glob.glob(os.path.join(INPUT_DIR, "*.zip")):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for csv_name in zf.namelist():
            if not csv_name.endswith(".csv"):
                continue

            chave = f"{os.path.basename(zip_path)}::{csv_name}"

            try:
                with zf.open(csv_name) as f:
                    header_line = ler_primeira_linha_seguro(f)
                    colunas = [col.strip().strip('"') for col in header_line.split(";")]

                    arquivo_colunas[chave] = set(colunas)
                    todas_colunas.update(colunas)

                    # Calcular a interseção de colunas presentes em todos os arquivos
                    if colunas_comuns is None:
                        colunas_comuns = set(colunas)
                    else:
                        colunas_comuns &= set(colunas)

            except Exception as e:
                print(f"❌ Erro ao processar {chave}: {e}")

# Ordenar colunas para consistência
todas_colunas = sorted(todas_colunas)
colunas_comuns = sorted(colunas_comuns)

# Construir a matriz
matriz = []
for arquivo, cols in arquivo_colunas.items():
    row = [1 if col in cols else 0 for col in todas_colunas]
    matriz.append([arquivo] + row)

# Criar o DataFrame sem a coluna de encoding
df = pd.DataFrame(matriz, columns=["arquivo_csv"] + todas_colunas)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

# Criar o arquivo com colunas comuns
if colunas_comuns:
    df_comum = pd.DataFrame(colunas_comuns, columns=["coluna_comum"])
    df_comum.to_csv(OUTPUT_COLUNAS_COMUNS_CSV, index=False)

print(f"✅ Matriz de colunas gerada: {OUTPUT_CSV}")
print(f"✅ Colunas comuns geradas: {OUTPUT_COLUNAS_COMUNS_CSV}")
print("colunas comuns:", colunas_comuns)
