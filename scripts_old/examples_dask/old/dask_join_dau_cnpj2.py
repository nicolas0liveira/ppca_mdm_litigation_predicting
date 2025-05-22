import time
import webbrowser

from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask

pbar = ProgressBar()
pbar.register()


def process_join():
    DATA_ROOT = '/home/nicolas/Documents/dados/'

    file_dau = f"{DATA_ROOT}/pgfn/intermediate/dau.parquet"
    file_cnpj = f"{DATA_ROOT}/cnpj/processed/cnpj.parquet"
    file_out = f"{DATA_ROOT}/pgfn/processed/dau_with_cnpj.parquet"

    print("🔹 Lendo dados de DAU...")
    dau_ddf = dd.read_parquet(file_dau)
    dau_ddf['cd_cnpj'] = dau_ddf['cd_cnpj'].astype('string')
    dau_ddf = dau_ddf.repartition(partition_size="100MB")

    print("🔹 Lendo dados de CNPJ...")
    cnpj_ddf = dd.read_parquet(file_cnpj, columns=["cd_cnpj", "cd_nivel1_secao"])
    cnpj_ddf['cd_cnpj'] = cnpj_ddf['cd_cnpj'].astype('string')
    cnpj_ddf = cnpj_ddf.repartition(npartitions=100)

    print("🔹 Realizando merge dos dados...")
    dau_completo = dau_ddf.merge(cnpj_ddf, how="left", on="cd_cnpj")

    print("🔹 Salvando dados combinados no disco...")
    # dau_completo.to_parquet(file_out, write_index=False, overwrite=True)

    print("✅ Pipeline finalizado com sucesso.")
    return 1


def tarefa_lenta(x):
    time.sleep(2)
    return x * x

def main():

    cluster = LocalCluster(dashboard_address=":9090")
    client = Client(cluster)

    print("Dask Client iniciado:")
    print(client)


    # Abre o dashboard no navegador
    webbrowser.open(client.dashboard_link)

    # Gera uma lista de tarefas Dask (lazy)
    tarefas = [dask.delayed(process_join)() for _ in range(1)]

    # Agrupa as tarefas e executa
    resultado = dask.compute(*tarefas)

    print("Resultados computados:")
    print(resultado)

    # Mantém o processo vivo para visualizar o dashboard
    input("Pressione Enter para encerrar...")

    # Fecha cliente e cluster
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
