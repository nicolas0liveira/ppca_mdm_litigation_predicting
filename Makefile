.PHONY: help install_env install pgfn-data start start-mdm stop clean clean_all clean_env
.DEFAULT_GOAL := help

help:
	@echo "Makefile para o projeto Dask"
	@echo "Comandos disponíveis:"
	@echo "  install       - cria/atualiza o ambiente e instala os requisitos"
	@echo "  start         - inicia o container Docker"
	@echo "  stop          - para o container Docker"
	@echo "  clean         - remove o container Docker"
	@echo "  clean_all     - remove o container e a imagem Docker"
	@echo "  clean_env     - remove o ambiente virtual Python"
	@echo "  pgfn-data     - baixa os dados da PGFN"
	@echo "  start-mdm     - inicia o container Docker com o MDM"

install_env: clean_env
	@echo "🔧 Criando ambiente virtual com python3.10"
	@python -m venv venv
	@echo "🔧 Atualizando pip"
	@./venv/bin/pip install --upgrade pip > /dev/null || (echo "Erro ao atualizar pip" && exit 1)
	@echo "🔧 Instalando dependências do requirements.txt"
	@./venv/bin/pip install -r requirements.txt > /dev/null || (echo "Erro ao instalar dependências" && exit 1)
	@echo "✅ Ambiente virtual criado com sucesso com python3.10"

install: install_env
	@echo "📁 Criando diretórios padrão do projeto..."
	@mkdir -p data/{sample,raw,interim,external,processed}
	@mkdir -p models reports notebooks docs references utils
	@mkdir -p scripts/config \
			  scripts/p01_data \
			  scripts/p02_etl \
			  scripts/p03_feature_engineering \
			  scripts/p04_modeling \
			  scripts/p05_evaluation \
			  scripts/p06_deployment \
			  scripts/p07_monitoring

	@for dir in config \
				utils \
				p01_data \
				p02_etl \
				p03_feature_engineering \
				p04_modeling \
				p05_evaluation \
				p06_deployment \
				p07_monitoring; do \
		touch scripts/$$dir/__init__.py; \
	done

	@touch scripts/__init__.py
	@touch scripts/config/settings.py

	@echo "DATA_DIR=$$HOME/Documents/dados" > example.env
	@echo "✅ Projeto inicializado com sucesso!"

pgfn-data:
	@echo "📥 Baixando dados do PGFN (simulado)"
	@./venv/bin/python scripts/p01_data/download_pgfn.py

start:
	@export DATASET_S3_PATH=s3://ppca/mdm/pgfn/processed/dau_so_com9atributos_importantes.parquet && ./venv/bin/python main_dask.py
	# @./venv/bin/python main_dask.py
	@echo "🚀 Iniciando"

start-mdm:
	@./venv/bin/python main_dask_mdm.py
	@echo "🚀 Iniciando"

stop:
	docker compose -f docker/dask/docker-compose.yml down

clean:
	docker compose -f docker/dask/docker-compose.yml rm -f

clean_all:
	docker compose -f docker/dask/docker-compose.yml down --rmi all

clean_env:
	@echo "🧹 Removendo ambiente virtual..."
	@rm -rf venv
	@find . -type d -name "__pycache__" -exec rm -r {} +
