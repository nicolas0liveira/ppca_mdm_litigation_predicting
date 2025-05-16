import dask.dataframe as dd
from dask.diagnostics import Profiler, ResourceProfiler, visualize, ProgressBar

DATA_ROOT = '/home/nicolas/Documents/dados/'

file_dau = f"{DATA_ROOT}/pgfn/intermediate/dau.parquet"
file_cnpj = f"{DATA_ROOT}/cnpj/processed/cnpj.parquet"
file_out = f"{DATA_ROOT}/pgfn/processed/dau_with_cnpj.parquet"

# Ativa barra de progresso global
pbar = ProgressBar()
pbar.register()

# 1. Leitura dos dados
print("🔹 Lendo dados de DAU...")
with Profiler() as prof1, ResourceProfiler() as rprof1:
    dau_ddf = dd.read_parquet(file_dau)

print("🔹 Lendo dados de CNPJ...")
with Profiler() as prof2, ResourceProfiler() as rprof2:
    cnpj_ddf = dd.read_parquet(file_cnpj, columns=["cd_cnpj", "cd_nivel1_secao"])

# # 2. set_index com shuffle="disk" (adiando execução com compute=False)
# print("🔹 Reparticionando DAU por CNPJ com shuffle='disk'...")
# with Profiler() as prof3, ResourceProfiler() as rprof3:
#     dau_ddf = dau_ddf.set_index("CNPJ", shuffle="disk", sorted=False, compute=False)

# print("🔹 Reparticionando CNPJ por cd_cnpj com shuffle='disk'...")
# with Profiler() as prof4, ResourceProfiler() as rprof4:
#     cnpj_ddf = cnpj_ddf.set_index("cd_cnpj", shuffle="disk", sorted=False, compute=False)

# 3. Join (adiado)
print("🔹 Realizando join dos dados...")
with Profiler() as prof5, ResourceProfiler() as rprof5:
    dau_completo = dau_ddf.join(cnpj_ddf, how="left")

# 4. Escrita dos dados (etapa que executa de fato)
print("🔹 Salvando dados combinados no disco...")
with Profiler() as prof6, ResourceProfiler() as rprof6, ProgressBar():
    dau_completo.to_parquet(file_out, write_index=False, overwrite=True)

# 5. Geração dos arquivos de diagnóstico
print("✅ Gerando arquivos de diagnóstico...")

visualize([prof1, rprof1], filename="01-leitura_dau_dask.html")
visualize([prof2, rprof2], filename="02-leitura_cnpj_dask.html")
# visualize([prof3, rprof3], filename="03-set_index_dau.html")
# visualize([prof4, rprof4], filename="04-set_index_cnpj.html")
# visualize([prof5, rprof5], filename="05-join_dask.html")
# visualize([prof6, rprof6], filename="06-save_output.html")

print("✅ Pipeline finalizado com sucesso.")
