from dask.distributed import Client, LocalCluster


def main():
    with LocalCluster(dashboard_address=":9090") as cluster:
        with Client(cluster) as client:
            print(client)
            input("Pressione Enter para encerrar...")

if __name__ == "__main__":
    main()
