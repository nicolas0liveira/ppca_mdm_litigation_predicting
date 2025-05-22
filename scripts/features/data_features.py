from typing import Tuple
import dask.dataframe as dd
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

from scripts.config.settings import RANDOM_STATE, TEST_SIZE

def get_preprocessor() -> ColumnTransformer:
    """
    Cria um ColumnTransformer para pré-processamento de dados numéricos e categóricos.

    Returns:
        ColumnTransformer: Transformador com escalonamento e codificação one-hot.
    """
    num_cols = ["idade_divida_dias"]
    cat_cols = [
        "tipo_devedor",
        "nm_porte",
        "origem",
        "situacao_inscricao",
        "nm_situacao_cadastral",
        "nm_regiao_politica",
    ]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def prepare_features(
    dataset: pd.DataFrame,
    fit_preprocessor: bool = True,
    preprocessor: ColumnTransformer = None
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Aplica pré-processamento ao dataset e retorna X, y e o preprocessor.
    Pode ser reutilizado para novos dados com o preprocessor já treinado.

    Args:
        dataset (pd.DataFrame): Dados de entrada com coluna 'target'.
        fit_preprocessor (bool): Se True, ajusta o preprocessor nos dados.
        preprocessor (ColumnTransformer): Transformador já ajustado (opcional).

    Returns:
        Tuple: X transformado, y, e preprocessor (ajustado ou reutilizado).
    """
    if isinstance(dataset, dd.DataFrame):
        dataset = dataset.compute()

    y = dataset["target"].str.upper().map({"SIM": 1, "NAO": 0, "NÃO": 0})
    X = dataset.drop(columns=["target", "numero_inscricao"])

    if fit_preprocessor or preprocessor is None:
        preprocessor = get_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    return X_transformed, y, preprocessor


def balance_data(X, y, threshold: float = 0.1, k_neighbors: int = 5):
    class_counts = Counter(y)
    print(f"Distribuição original das classes: {class_counts}")

    if len(class_counts) != 2:
        print("⚠️ Função projetada para problemas binários.")
        return X, y

    major_class = max(class_counts, key=class_counts.get)
    minor_class = min(class_counts, key=class_counts.get)

    major_count = class_counts[major_class]
    minor_count = class_counts[minor_class]

    imbalance_ratio = abs(major_count - minor_count) / (major_count + minor_count)

    if imbalance_ratio > threshold:
        print(f"⚠️ Classes desbalanceadas (razão: {imbalance_ratio:.2f}). Aplicando SMOTE...")
        try:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print(f"✅ Após SMOTE: {Counter(y_resampled)}")
            return X_resampled, y_resampled
        except ValueError as e:
            print(f"❌ Erro ao aplicar SMOTE: {e}")
            print("Voltando aos dados originais.")
            return X, y
    else:
        print("✅ Classes estão balanceadas. SMOTE não necessário.")
        return X, y

def split_train_test(
    dataset: pd.DataFrame,
    preprocessor: ColumnTransformer = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Divide o dataset em treino/teste e aplica pré-processamento e balanceamento.

    Args:
        dataset (pd.DataFrame): DataFrame contendo os dados brutos.
        preprocessor (ColumnTransformer, opcional): Se já existir, será reutilizado.

    Returns:
        Tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    if isinstance(dataset, dd.DataFrame):
        dataset = dataset.compute()

    y = dataset["target"].str.upper().map({"SIM": 1, "NAO": 0, "NÃO": 0})
    X = dataset.drop(columns=["target", "numero_inscricao"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    if preprocessor is None:
        preprocessor = get_preprocessor()
        X_train = preprocessor.fit_transform(X_train)
    else:
        X_train = preprocessor.transform(X_train)

    X_test = preprocessor.transform(X_test)

    X_train, y_train = balance_data(X_train, y_train)

    return X_train, X_test, y_train, y_test, preprocessor



# # Treinamento
# X_train, X_test, y_train, y_test, preprocessor = split_train_test(train_df)

# # Aplicação em novos dados
# X_new, y_new, _ = prepare_features(new_df, fit_preprocessor=False, preprocessor=preprocessor)
# preds = model.predict(X_new)