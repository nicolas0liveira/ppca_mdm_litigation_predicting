import dask.dataframe as dd


def feature_engineering(ddf):
    """
    Realiza a engenharia de features no DataFrame Dask.
    Args:
        ddf (dd.DataFrame): DataFrame Dask a ser processado.
    Returns:
        dd.DataFrame: DataFrame Dask com as features processadas.
    """
    # Exemplo de engenharia de features
    # ddf["idade_divida_dias"] = ddf["idade"] / 365
    # ddf["tipo_devedor"] = ddf["tipo_devedor"].str.lower()
    # ddf["tipo_devedor"] = ddf["tipo_devedor"].str.replace(".", "_")
    return ddf
