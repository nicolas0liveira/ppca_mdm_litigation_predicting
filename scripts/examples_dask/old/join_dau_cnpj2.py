
import dask.dataframe as dd
DATA_ROOT = '/home/nicolas/Documents/dados/'

# Caminhos
file_dau = f"{DATA_ROOT}/pgfn/intermediate/dau.parquet"
file_cnpj = f"{DATA_ROOT}/cnpj/processed/cnpj.parquet"
file_out = f"{DATA_ROOT}/pgfn/processed/dau_with_cnpj.parquet"

# 1. Lê os dados
dau_ddf = dd.read_parquet(file_dau)
cnpj_ddf = dd.read_parquet(file_cnpj)

# 2. Renomeia a coluna da base cnpj para permitir o join
cnpj_ddf = cnpj_ddf.rename(columns={"cd_cnpj": "CNPJ"})

# 3. Faz o join diretamente sem coletar lista de CNPJs
dau_completo = dau_ddf.merge(cnpj_ddf, on="CNPJ", how="left")

# 4. (Opcional) particiona por CNPJ para salvar
dau_completo = dau_completo.set_index("CNPJ")
dau_completo.to_parquet(file_out, write_index=True, overwrite=True)
