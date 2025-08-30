#!/bin/bash

# Define o nome do arquivo de dados OSRM
OSRM_DATA_FILE="sudeste-latest.osrm"
OSRM_REGION=south-america/brazil
OSRM_PBF_FILE="sudeste-latest.osm.pbf"
OSRM_LINK_DOWNLOAD_PBF_FILE="https://download.geofabrik.de/$OSRM_REGION/sudeste-latest.osm.pbf"

# Pasta onde os dados OSRM são armazenados
DATA_DIR="data"

echo "Verificando o status da inicialização do OSRM..."

# Verifica se a pasta de dados existe, senão, a cria.
if [ ! -d "$DATA_DIR" ]; then
    mkdir "$DATA_DIR"
    echo "Pasta '$DATA_DIR' criada."
fi

# Verifica se o arquivo .osrm já foi processado
if [ -f "$DATA_DIR/$OSRM_DATA_FILE" ]; then
    echo "O arquivo '$OSRM_DATA_FILE' já existe. Iniciando o servidor de roteamento."
    # Apenas inicia o serviço de roteamento
    docker-compose up -d osrm-routed
else
    echo "Arquivo '$OSRM_DATA_FILE' não encontrado. Iniciando o processo de construção."

    # Verifica se o arquivo .pbf existe, senão, o baixa
    if [ ! -f "$DATA_DIR/$OSRM_PBF_FILE" ]; then
        echo "Baixando o arquivo de dados .pbf..."
        wget $OSRM_LINK_DOWNLOAD_PBF_FILE -P "$DATA_DIR"
    fi

    echo "Iniciando o container de construção..."
    docker-compose up -d osrm-builder

    sleep 5 # Dá um tempo para o container builder iniciar

    # --- Executa as etapas de processamento em ordem ---
    echo "1/3. Executando osrm-extract..."
    docker-compose exec osrm-builder osrm-extract -p /opt/car.lua /data/"$OSRM_PBF_FILE"

    echo "2/3. Executando osrm-partition..."
    docker-compose exec osrm-builder osrm-partition /data/"$OSRM_DATA_FILE"

    echo "3/3. Executando osrm-customize..."
    docker-compose exec osrm-builder osrm-customize /data/"$OSRM_DATA_FILE"

    echo "Processo de construção concluído! Parando o container builder..."
    docker-compose down osrm-builder

    echo "Iniciando o servidor de roteamento..."
    docker-compose up -d osrm-routed
fi

echo "Inicialização concluída. O servidor OSRM deve estar rodando em http://localhost:5000"