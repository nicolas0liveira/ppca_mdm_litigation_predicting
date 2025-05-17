import pandas as pd
import os
import argparse

def split_parquet(input_path, output_dir):
    # Carrega o arquivo Parquet inteiro
    print(f"Lendo arquivo: {input_path}")
    df = pd.read_parquet(input_path)

    # Divide o DataFrame em 3 partes
    parts = [df.iloc[i::3].reset_index(drop=True) for i in range(3)]

    # Garante que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)

    # Salva cada parte em um novo arquivo Parquet
    for idx, part in enumerate(parts):
        output_path = os.path.join(output_dir, f"part_{idx+1}.parquet")
        part.to_parquet(output_path, index=False)
        print(f"Salvo: {output_path} ({len(part)} linhas)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide um arquivo Parquet em 3 partes")
    parser.add_argument("input_path", help="Caminho do arquivo .parquet original")
    parser.add_argument("output_dir", help="Diretório onde os arquivos divididos serão salvos")

    args = parser.parse_args()
    split_parquet(args.input_path, args.output_dir)
