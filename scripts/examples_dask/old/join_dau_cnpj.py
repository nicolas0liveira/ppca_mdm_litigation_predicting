#Usando dask

import dask.dataframe as dd
DATA_ROOT = '/home/nicolas/Documents/dados/'
# DATA_ROOT = DATA_ROOT if DATA_ROOT else '/home/nicolas/Documents/dados/'

# Caminhos
file_dau = f"{DATA_ROOT}/pgfn/intermediate/dau.parquet"
file_cnpj = f"{DATA_ROOT}/cnpj/processed/cnpj.parquet"
file_out = f"{DATA_ROOT}/pgfn/processed/dau_with_cnpj.parquet"

# 1. Lê as bases com Dask
dau_ddf = dd.read_parquet(file_dau)
cnpj_ddf = dd.read_parquet(file_cnpj)

# 2. Coleta os CNPJs únicos da base DAU
cnpjs_necessarios = dau_ddf["CNPJ"].dropna().unique().compute().tolist()

# 3. Filtra a base CNPJ
cnpj_filtrado = cnpj_ddf[cnpj_ddf["cd_cnpj"].isin(cnpjs_necessarios)]

# 4. Renomeia a coluna para permitir o join
cnpj_filtrado = cnpj_filtrado.rename(columns={"cd_cnpj": "CNPJ"})

# 5. Faz o join com a base DAU
dau_completo = dau_ddf.merge(cnpj_filtrado, on="CNPJ", how="left")

# 6. Salva o resultado em Parquet ordenado por CNPJ e NUMERO_INSCRICAO
# OBS: Dask não garante ordenação durante o write. Ordenamos antes com `set_index` se for crucial.
dau_completo = dau_completo.set_index("CNPJ")  # Indexa por CNPJ para particionar
dau_completo.to_parquet(file_out, write_index=True, overwrite=True)

# Se quiser ver o resultado após computar
# resultado_df = dau_completo.compute()
# resultado_df.head()
