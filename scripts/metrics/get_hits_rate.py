from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd

def get_hits_in_data(
    dataset: pd.DataFrame,
    preprocessor,
    model,
    target_label: str = "SIM"
):
    """
    Aplica o modelo a um dataset, calcula a taxa de acerto para a classe desejada
    e retorna a matriz de confusão.

    Args:
        dataset (pd.DataFrame or dd.DataFrame): Novo conjunto de dados com coluna 'target'.
        preprocessor (ColumnTransformer): Pré-processador treinado.
        model: Modelo treinado (com método .predict()).
        target_label (str): Classe que será considerada como acerto (ex: "SIM").

    Returns:
        Tuple[float, ConfusionMatrixDisplay]: Taxa de acerto e display da matriz de confusão.
    """

    if isinstance(dataset, dd.DataFrame):
        dataset = dataset.compute()

    y_true = dataset["target"].str.upper().map({"SIM": 1, "NAO": 0, "NÃO": 0})
    X = dataset.drop(columns=["target", "numero_inscricao"])
    X_transformed = preprocessor.transform(X)
    y_pred = model.predict(X_transformed)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Calcular taxa de acerto (para a classe positiva)
    target_bin = 1 if target_label.upper() == "SIM" else 0
    hits = (y_pred == y_true) & (y_true == target_bin)
    total_target = (y_true == target_bin).sum()

    hits_rate = hits.sum() / total_target if total_target > 0 else 0.0

    return hits_rate, disp
