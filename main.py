from datetime import datetime
import time
from pathlib import Path
import os
import argparse
import webbrowser
from distributed import LocalCluster
import joblib

from loguru import logger
from matplotlib import pyplot as plt
from dask.distributed import Client

 # Models
# Decision Tree
from scripts.models.decision_tree.train_dt import train_dt, train_dt_dask

# # Logistic Regression
from scripts.models.logistic_regression.train_lr import train_lr, train_lr_dask

# # Métricas
from scripts.metrics.get_hits_rate import get_hits_in_data
from scripts.features.features import feature_engineering
from scripts.data.dataset import load_dataset
from scripts.features.data_features import split_train_test

def process(dask_client=None):
    dataset = load_dataset(frac=1, sample=False, compute=True)

    # Engenharia de features
    logger.info('Tratando dados...')

    dataset = feature_engineering(dataset)

    logger.info('Separando treino e teste...')
    X_train, X_test, y_train, y_test, preprocessor = split_train_test(dataset)

    # # Treinar o modelos simples
    if dask_client:
        # Logistic Regression
        logger.info('Treinando o modelo de regressão logística com Dask...')
        model_lr, f1_score_lr = train_lr_dask(X_train, y_train, X_test, y_test, dask_client)

        # Decision Tree
        logger.info('Treinando o modelo de árvore de decisão com dask...')
        model_dt, f1_score_dt = train_dt_dask(X_train, y_train, X_test, y_test, dask_client)

    else:
        logger.info('Treinando o modelo de regressão logística...')
        model_lr, f1_score_lr = train_lr(X_train, y_train, X_test, y_test)
        logger.info('Treinando o modelo de árvore de decisão...')
        model_dt, f1_score_dt = train_dt(X_train, y_train, X_test, y_test)



    # Armazenar os modelos e seus f1_scores em um dicionário
    models = {
        'logistic_regression': (model_lr, f1_score_lr),
        'decision_tree': (model_dt, f1_score_dt),
        # 'bagging': (model_bagging, f1_score_bagging),
        # 'boosting': (model_boosting, f1_score_boosting),
        # 'stacking': (model_stacking, f1_score_stacking),
        # 'voting_hard': (model_voting_hard, f1_score_voting_hard),
        # 'voting_soft': (model_voting_soft, f1_score_voting_soft)
    }

    logger.info('Comparando os modelos...')
    # models = {name: (model, f1_score) for name, (model, f1_score) in models.items()}
    # for name, (model, f1_score) in models.items():
    #     logger.info(f'Modelo: {name}, f1_score: {f1_score:.2f}')

    # logger.info('Encontrando o melhor modelo...')
    best_model_name, (best_model, best_f1_score) = max(models.items(), key=lambda item: item[1][1])
    # logger.info(f'O melhor modelo é {best_model_name} com um f1_score de {best_f1_score:.2f}')

    # # Fazer as predições e calcular métricas
    # logger.info('Fazendo predições e calculando métricas...')
    # f1, bce = make_predictions(X_train, X_test, y_train, y_test, best_model)
    # logger.info(f'F1 Score: {f1:.2f}')
    # logger.info(f'Binary Cross Entropy: {bce:.2f}')

    # Criar diretório dos artefatos gerados
    logger.info('Criando diretório dos artefatos gerados...')
    current_date = datetime.now().strftime('%Y%m%d%H%M')
    artefact_path = os.path.join('.', 'output', f'artefatos_{current_date}')
    Path(artefact_path).mkdir(parents=True, exist_ok=True)

    # Calcular  taxa de acerto e criando matriz de confusao
    logger.info('Calculando taxa de acerto e criando matriz de confusão...')
    target = 'SIM'
    hits_rate, disp = get_hits_in_data(
        dataset,
        preprocessor,
        best_model,
        target
    )
    logger.info(f'Taxa de acerto do melhor modelo ({best_model_name}): {hits_rate:.2f}')
    disp.plot()
    plt.savefig(os.path.join(artefact_path, f'confusion_matrix_{best_model_name}.png'))
    plt.show()
    plt.close()

    params = best_model.get_params()
    logger.info(f'Parâmetros do melhor modelo ({best_model_name}):')
    for param, value in params.items():
        logger.info(f'\t{param}: {value}')

    # Salvar o modelo e o preprocessador localmente
    logger.info('Salvando modelo e preprocessador...')
    joblib.dump(best_model, os.path.join(artefact_path, 'model.pkl'))
    joblib.dump(preprocessor, os.path.join(artefact_path, 'preprocessor.pkl'))

    logger.info('Salvando relatório em um arquivo de texto...')
    with open(os.path.join(artefact_path, 'relatorio.txt'), 'w', encoding='utf-8') as f:
        for model_name, (model, f1_score) in models.items():
            f.write(f'{model_name}: {f1_score}\n')
        f.write('\n\n-----------------------------------\n')
        f.write(f'CHAMPION: {best_model_name}')
        f.write('\n-------------------------------------\n')
        f.write(f'Melhor f1_score: {best_f1_score}\n')
        f.write(f'Porcentagem de acertos: {hits_rate:.2f}\n')
        f.write('Parâmetros do melhor modelo\n')
        for param, value in params.items():
            f.write(f'\t {param}: {value}\n')
        


def parse_args():
    parser = argparse.ArgumentParser(description="Executa o pipeline de ML com Dask")
    parser.add_argument("--execute", action="store_true", help="Executa o pipeline (senão só gera a DAG)")
    parser.add_argument("--keep-alive", action="store_true", help="Mantém o Dask Dashboard aberto após execução")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    client = None
    cluster = None

    try:
        logger.info("Iniciando pipeline ...")

        # Inicializa o Dask
        # client = Client("tcp://dask-scheduler:8786")
        cluster = LocalCluster(
            n_workers=4,
            threads_per_worker=2,
            memory_limit="8GB",
            dashboard_address=":9090",
            local_directory="./dask-worker-space",
        )
        client = Client(cluster)
        logger.info("Dask Client iniciado:")
        logger.info(client)

        webbrowser.open("http://localhost:9090")
        logger.info("Aguardando 5 segundos para o dashboard carregar...")
        time.sleep(5)
        logger.info(f"Dask Dashboard disponível em: {client.dashboard_link}")

        # Executa o pipeline principal
        process(client)

        if args.keep_alive:
            if args.keep_alive:
                input("Pressione ENTER para encerrar e fechar o dashboard...")


        logger.info("Pipeline finalizado com sucesso.")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrompido pelo usuário (CTRL+C).")

    except SystemExit:
        logger.info("Pipeline encerrado pelo sistema.")

    except Exception as e:
        logger.exception(f"Erro durante execução do pipeline: {e}")
        # logger.debug(traceback.format_exc())  # Opcional: detalhamento completo

    finally:
        if client:
            client.close()
            logger.info("Dask Client fechado.")
        if cluster:
            cluster.close()
            logger.info("Dask Cluster fechado.")






# try:
#         run_pipeline(
#             parquet_s3path="s3://ppca/mdm/pgfn/processed/dau_so_com9atributos_importantes_50NAOe50SIM.parquet",
#             target_column="target",
#             experiment_name="logistic_regression_optuna",
#             client=client,
#             execute=args.execute
#         )
#     except Exception as e:
#         logger.error(f"Erro durante execução do pipeline: {e}")
#         client.close()