from dask.distributed import Client, LocalCluster
import dask.array as da

# Cria um cluster local com IP 127.0.0.1
cluster = LocalCluster(ip='127.0.0.1', n_workers=2, threads_per_worker=2, memory_limit='512MB')

# Conecta o client ao cluster
client = Client(cluster)

print("Dashboard:", client.dashboard_link)

# Cria um array Dask com 1 bilhão de elementos (dividido em chunks)
x = da.random.random(size=(1_000_000,), chunks=(100_000,))

# Operação paralela: média dos valores
result = x.mean()

# Computa o resultado
mean_value = result.compute()

print("Média dos valores:", mean_value)

# Encerra o client
client.close()
cluster.close()
