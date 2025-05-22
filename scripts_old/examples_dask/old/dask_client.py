from dask.distributed import Client

# client = Client("tcp://localhost:8786)
client = Client()
                
print(client)

# Mostra o link do dashboard
print(client.dashboard_link)
